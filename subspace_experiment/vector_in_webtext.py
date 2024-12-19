# %% [markdown]
# # Neural Network Activation Analysis
# This notebook analyzes neural network activations in language models, focusing on concept representations in hidden states.

# %% [markdown]
# ## Configuration and Imports

# %%
import torch
import numpy as np
from IPython.display import display, HTML
import einops
import datasets
from nnsight import LanguageModel

# %% [markdown]
# ## Experimental Configuration

# %%
# Model Configuration
MODEL_NAME = "google/gemma-2-2b"
MODEL_CACHE_DIR = "/share/u/can/models"
DEVICE = torch.device("cuda:0")

# Analysis Configuration
LAYERS_TO_ANALYZE = [12]
NUM_ENTITIES = 3
BATCH_SIZE = 50
NUM_BATCHES = 20
NUM_SINGULAR_VECTORS = 2
NUM_VISUALIZATIONS_PER_VECTOR = 5

# OpenWebText Configuration
WEBTEXT_DATASET = "Skylion007/openwebtext"
WEBTEXT_BATCH_SIZE = 5
WEBTEXT_NUM_BATCHES = 10
MAX_SEQ_LENGTH = 64

# Tracer Configuration
TRACER_KWARGS = {"scan": False, "validate": False}

# %% [markdown]
# ## Visualization Functions


# %%
def get_color_gradient(value, vmin, vmax):
    """
    Convert value to RGB color string:
    negative -> red
    zero -> white
    positive -> blue
    """
    abs_max = max(abs(vmin), abs(vmax))
    normalized = value / abs_max if abs_max != 0 else 0

    if normalized < 0:  # Red for negative
        r = 255
        g = b = int(255 * (1 + normalized))
    else:  # Blue for positive
        b = 255
        r = g = int(255 * (1 - normalized))

    return f"rgb({r}, {g}, {b})"


def get_text_color(value, abs_max):
    """Return black for abs(value) < 0.5, white for abs(value) >= 0.5"""
    return "white" if abs(value) >= 0.5 * abs_max else "black"


def visualize_token_activations(tokens, activations):
    """
    Visualize tokens as colored inline rectangles with their activation values.
    """
    if isinstance(activations, torch.Tensor):
        activations = activations.detach().cpu().numpy()

    if len(tokens) != len(activations):
        raise ValueError(
            f"Length mismatch: {len(tokens)} tokens vs {len(activations)} activation values"
        )

    vmin, vmax = np.min(activations), np.max(activations)
    abs_max = max(abs(vmin), abs(vmax))

    scale_ref = f"""
        <div style='margin-bottom: 20px;'>
            <span style='margin-right: 15px; font-size: 0.9em;'>
                Scale: 
                <span style='background-color: {get_color_gradient(vmin, vmin, vmax)}; 
                           color: {get_text_color(vmin, abs_max)}; 
                           padding: 2px 8px; 
                           border-radius: 3px;'>
                    min: {vmin:.3f}
                </span>
                <span style='background-color: white; 
                           color: black; 
                           padding: 2px 8px; 
                           border-radius: 3px; 
                           margin: 0 5px;'>
                    0.000
                </span>
                <span style='background-color: {get_color_gradient(vmax, vmin, vmax)}; 
                           color: {get_text_color(vmax, abs_max)}; 
                           padding: 2px 8px; 
                           border-radius: 3px;'>
                    max: {vmax:.3f}
                </span>
            </span>
        </div>
    """

    tokens_html = ""
    for token, value in zip(tokens, activations):
        bg_color = get_color_gradient(value, vmin, vmax)
        text_color = get_text_color(value, abs_max)

        tokens_html += f"""
            <span class='token-box' style='background-color: {bg_color}; color: {text_color};'>
                <span class='activation-value' style='color: #666;'>{value:.3f}</span>
                {token}
            </span>
        """

    html_output = f"""
    <div style='font-family: monospace; line-height: 2.5; background-color: white; padding: 20px;'>
        <style>
            .token-box {{
                display: inline-block;
                padding: 4px 8px;
                margin: 2px;
                border-radius: 3px;
                font-weight: bold;
                position: relative;
            }}
            .activation-value {{
                position: absolute;
                top: -18px;
                left: 50%;
                transform: translateX(-50%);
                font-size: 0.8em;
                white-space: nowrap;
            }}
        </style>
        {scale_ref}
        {tokens_html}
    </div>
    """

    display(HTML(html_output))


# %% [markdown]
# ## Model Setup

# %%
# Initialize model
model = LanguageModel(
    MODEL_NAME,
    dispatch=True,
    cache_dir=MODEL_CACHE_DIR,
    device_map=DEVICE,
)

# %% [markdown]
# ## Data Processing

# %%
V_singular_vectors = torch.load("V_singular_vectors.pt")
V_singular_vectors.shape


# %% [markdown]
# ## OpenWebText Analysis
# %%
# ds_web = datasets.load_dataset(WEBTEXT_DATASET, split="train", streaming=True)

import json

with open("dataset.json", "r") as f:
    ds_web = json.load(f)


# Load OpenWebText dataset
def tokenized_batches(ds_web, batch_size, num_batches, max_length=128):
    ds_web = iter(ds_web)
    batches = []
    for i in range(num_batches):
        batch = [next(ds_web)["text"] for _ in range(batch_size)]
        tokenized = model.tokenizer.batch_encode_plus(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            padding_side="right",
        )
        batches.append(tokenized)
    return batches


web_token_id_batches = tokenized_batches(
    ds_web, WEBTEXT_BATCH_SIZE, WEBTEXT_NUM_BATCHES, MAX_SEQ_LENGTH
)

# Extract activations from OpenWebText
web_acts_bBLD = torch.zeros(
    (WEBTEXT_NUM_BATCHES, WEBTEXT_BATCH_SIZE, MAX_SEQ_LENGTH, model.config.hidden_size)
)

for batch_idx, token_ids in enumerate(web_token_id_batches):
    token_ids = token_ids.to(DEVICE)
    with torch.no_grad(), model.trace(token_ids, **TRACER_KWARGS):
        for layer in LAYERS_TO_ANALYZE:
            web_acts_bBLD[batch_idx] = model.model.layers[layer].output[0].save()

torch.cuda.empty_cache()

# Process OpenWebText activations
web_acts_ALD = einops.rearrange(web_acts_bBLD, "b B L D -> (b B) L D")
# normalize along final dimension
web_acts_ALD = web_acts_ALD / torch.norm(web_acts_ALD, dim=-1, keepdim=True)

web_token_AL = torch.zeros(
    (WEBTEXT_NUM_BATCHES * WEBTEXT_BATCH_SIZE, MAX_SEQ_LENGTH), dtype=torch.long
)

for i, token_ids in enumerate(web_token_id_batches):
    start_idx = i * WEBTEXT_BATCH_SIZE
    end_idx = (i + 1) * WEBTEXT_BATCH_SIZE
    web_token_AL[start_idx:end_idx] = token_ids["input_ids"]

web_str_AL = [
    [model.tokenizer.decode(token_id) for token_id in current_token_ids]
    for current_token_ids in web_token_AL
]

# %% [markdown]
# ## Visualization of Results

# %%
# Visualize activations along V directions
for v_index in range(NUM_SINGULAR_VECTORS):
    data_along_pc = web_acts_ALD @ V_singular_vectors[:, v_index].to(web_acts_ALD.device)

    print("##################")
    print(f"Singular vector V {v_index}")
    print("##################")

    for i in range(len(web_str_AL)):
        visualize_token_activations(web_str_AL[i], data_along_pc[i])
        if i > NUM_VISUALIZATIONS_PER_VECTOR:
            break

# %%
