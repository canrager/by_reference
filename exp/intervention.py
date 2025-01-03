# %%
import os
import torch
import json
from tqdm import tqdm

import numpy as np
import einops
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.activation import load_model, get_activation_cache_for_entity
from utils.dataset import generate_base_source_dataset
from utils.generation import *

torch.set_grad_enabled(False)

# %%
model_id = "google/gemma-2-2b"
model = load_model(model_id)
model.tokenizer.pad_token = model.tokenizer.eos_token

# %% [markdown]
# ## Cache activations

# %%
# Example usage
num_samples_per_order_pair = 53  # post filter
random_offset = True
ctx_length_with_pad = 100
layers = range(model._model.config.num_hidden_layers)

template_name = "box"
base_entity_name = "common_objects1"
source_entity_name = "common_objects2"
attribute_name = "letter"

# template_name = "country"
# base_entity_name = "first_names1"
# source_entity_name = "first_names2"
# attribute_name = "country"

# template_name = "month"
# base_entity_name = "first_names1"
# source_entity_name = "first_names2"
# attribute_name = "month"

force_recompute_data = False
force_recompute_acts = False

# %%
dataset_path = f"data/base_source_dataset_{template_name}_spo{num_samples_per_order_pair}.json"
activation_path = f"data/source_activations_{template_name}_spo{num_samples_per_order_pair}.pt"

if force_recompute_data or not os.path.exists(dataset_path):
    dataset_dict = generate_base_source_dataset(
        model,
        num_samples_per_order_pair,
        template_name=template_name,
        base_entity_name=base_entity_name,
        source_entity_name=source_entity_name,
        attribute_name=attribute_name,
        tokenizer=model.tokenizer,
        ctx_length_with_pad=ctx_length_with_pad,
        random_offset=random_offset,
        save_path=dataset_path,
    )
else:
    with open(dataset_path, "r") as f:
        dataset_dict = json.load(f)

base_dataset = dataset_dict["base"]
source_dataset = dataset_dict["source"]

# Get activations for source dataset
if force_recompute_acts or not os.path.exists(activation_path):
    act_LBED, pos_LE = get_activation_cache_for_entity(
        model,
        source_dataset,
        layers,
        save_activations_path=activation_path,
        save_positions_path=None,
    )
else:
    with open(activation_path, "rb") as f:
        act_LBED = torch.load(f)


# %%
# Prepare the activations

with open(activation_path, "rb") as f:
    act_LBED = torch.load(f)

# Get relation orders
source_relation_orders = [int(source["question"]["answer_key"][1:]) for source in source_dataset]
base_relation_orders = [int(base["question"]["answer_key"][1:]) for base in base_dataset]

# Sort activations by relation
num_relations = len(set(source_relation_orders))
num_samples = len(source_dataset)
num_samples_per_relation = int(num_samples / num_relations)

act_LRBD = torch.zeros(
    (len(layers), num_relations, num_samples_per_relation, model._model.config.hidden_size)
)
for relation_idx in range(num_relations):
    mask = torch.tensor(source_relation_orders) == relation_idx
    act_LRBD[:, relation_idx] = act_LBED[:, mask, -1, :]

act_LRBD.shape


# %% [markdown]
# ### Single SVD component


from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import torch
import einops
from tqdm import tqdm
from IPython.display import display, HTML


def plot_singular_vectors_interactive(
    act_LRBD, num_vs, layers, template_name, save_path=None, fixed_svd_layer=None
):
    """
    Plot singular vectors with optional fixed SVD from a specific layer.

    Args:
        act_LRBD: Activation tensor
        num_vs: Number of singular vectors to plot
        layers: List of layer indices
        template_name: Name of the task/template
        save_path: Path to save the HTML plot
        fixed_svd_layer: Layer index to use for fixed SVD basis (if None, compute SVD per layer)
    """
    num_relations = act_LRBD.shape[1]
    num_samples_per_relation = act_LRBD.shape[2]

    titles = [f"n={v}" for v in range(num_vs)]
    fig = make_subplots(rows=1, cols=num_vs, subplot_titles=titles)

    colors = px.colors.qualitative.Set1[:num_relations]

    # Compute fixed SVD if specified
    fixed_V = None
    if fixed_svd_layer is not None:
        if fixed_svd_layer not in layers:
            raise ValueError(f"fixed_svd_layer {fixed_svd_layer} must be in layers {layers}")
        act_BD = einops.rearrange(act_LRBD[fixed_svd_layer], "b r d -> (b r) d")
        _, _, fixed_V = torch.svd(act_BD)

    svd_results = []
    for layer_idx in tqdm(range(len(layers)), desc="Computing projections"):
        act_BD = einops.rearrange(act_LRBD[layer_idx], "b r d -> (b r) d")

        if fixed_V is None:
            _, _, V = torch.svd(act_BD)
        else:
            V = fixed_V

        svd_results.append(V)

    for layer_idx, layer in enumerate(layers):
        V = svd_results[layer_idx]

        for v in range(num_vs):
            for i in range(num_relations):
                pointer_components = act_LRBD[layer_idx][i] @ V.T[v]

                fig.add_trace(
                    go.Scatter(
                        x=torch.arange(num_samples_per_relation).numpy(),
                        y=pointer_components.numpy(),
                        mode="markers",
                        name=f"Pointer {i}",
                        legendgroup=f"Pointer {i}",
                        marker=dict(color=colors[i]),
                        showlegend=(v == 0),
                        visible=layer_idx == 0,
                    ),
                    row=1,
                    col=v + 1,
                )

            fig.update_xaxes(
                title_text="Samples" if v == num_vs // 2 else None, row=1, col=v + 1
            )
            fig.update_yaxes(
                title_text="V[n] @ resid_activations" if v == 0 else None, row=1, col=v + 1
            )

    steps = []
    for layer_idx in range(len(layers)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            label=f"Layer {layers[layer_idx]}"
            + (" (reference)" if layers[layer_idx] == fixed_svd_layer else ""),
        )
        for v in range(num_vs):
            for i in range(num_relations):
                trace_idx = layer_idx * (num_vs * num_relations) + v * num_relations + i
                step["args"][0]["visible"][trace_idx] = True
        steps.append(step)

    title_text = f"Projection of resid onto singular vector n, Task: {template_name}"
    if fixed_svd_layer is not None:
        title_text += f" (Fixed SVD from layer {fixed_svd_layer})"
    title_text += f"\nUse arrow keys to navigate layers; click on legend to toggle traces."

    fig.update_layout(
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Layer: "},
            pad={"t": 50},
            steps=steps,
        )],
        title=title_text,
        height=500,
        showlegend=True,
    )

    # Improved arrow key navigation JavaScript
    arrow_key_js = """
    <script>
    (function() {
        function setupKeyboardNavigation() {
            // Find the Plotly div - more reliable selector
            var plotDivs = document.getElementsByClassName('plotly-graph-div');
            if (plotDivs.length === 0) {
                // If not found, try again after a short delay
                setTimeout(setupKeyboardNavigation, 100);
                return;
            }
            
            var plotDiv = plotDivs[0];
            
            function handleArrowKey(event) {
                if (!event.target.tagName.match(/input|textarea/i)) {  // Ignore if typing in input/textarea
                    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                        event.preventDefault();  // Prevent page scrolling
                        
                        var slider = plotDiv._fullLayout.sliders[0];
                        var currentStep = slider.active || 0;
                        var numSteps = slider.steps.length;
                        
                        // Calculate new step
                        var newStep = currentStep + (event.key === 'ArrowRight' ? 1 : -1);
                        newStep = Math.max(0, Math.min(newStep, numSteps - 1));
                        
                        if (newStep !== currentStep) {
                            // Get visibility settings from the step
                            var update = slider.steps[newStep].args[0].visible;
                            
                            // Update both the slider position and trace visibility
                            Plotly.update(plotDiv, 
                                {visible: update},  // Update trace visibility
                                {'sliders[0].active': newStep}  // Update slider position
                            );
                        }
                    }
                }
            }
            
            // Remove any existing listeners to prevent duplicates
            document.removeEventListener('keydown', handleArrowKey);
            // Add the event listener
            document.addEventListener('keydown', handleArrowKey);
        }
        
        // Initialize keyboard navigation
        if (document.readyState === 'complete') {
            setupKeyboardNavigation();
        } else {
            window.addEventListener('load', setupKeyboardNavigation);
        }
    })();
    </script>
    """

    if save_path is not None:
        # Convert figure to HTML and add the arrow key navigation
        fig_html = fig.to_html(full_html=True, include_plotlyjs=True)
        # Insert arrow key JavaScript before the closing </body> tag
        fig_html = fig_html.replace("</body>", f"{arrow_key_js}</body>")
        with open(save_path, "w") as f:
            f.write(fig_html)
    else:
        # For notebook display
        display(HTML(arrow_key_js))
        fig.show()

    return fig


reference_layer = 12

# num_vs = 5
# save_path = f"images/singular_vectors_{num_vs}_{template_name}_spo{num_samples_per_order_pair}.html"

# plot_singular_vectors_interactive(
#     act_LRBD,
#     num_vs,
#     layers,
#     template_name,
#     save_path=save_path,
#     fixed_svd_layer=reference_layer,
# )


# %% [markdown]
# ### Single SVD component with mean, task generalization

# %%

def plot_generalization(
    v_outcomes_ds, labels, pointer_component_dims, save_path=None
):
    # Separate singular and random outcomes
    singular_outcomes = {k: v["singular"] for k, v in v_outcomes_ds.items()}
    random_outcomes = {k: v["random"] for k, v in v_outcomes_ds.items()}

    # Normalize other_base_object by 3
    for k in singular_outcomes.keys():
        singular_outcomes[k][labels.index("other_objects") - 1] /= 3
        random_outcomes[k][labels.index("other_objects") - 1] /= 3

    # Calculate percentages
    proportions = {k: v / v.sum() for k, v in singular_outcomes.items()}
    random_proportions = {k: v / v.sum() for k, v in random_outcomes.items()}

    # Set up the plot with adjusted figure size to accommodate external legend
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)

    # Calculate bar positions
    n_groups = len(proportions.keys())
    n_bars = len(labels)
    group_width = 0.8
    bar_width = group_width / n_bars

    # Create bars grouped by k
    random_line_added_to_legend = False
    for i, label in enumerate(labels):
        positions = np.arange(n_groups) + i * bar_width
        values = [proportions[k][i] for k in proportions.keys()]
        random_values = [random_proportions[k][i] for k in random_proportions.keys()]

        bars = ax.bar(positions, values, bar_width, label=label)
        # # Add random baseline lines
        # for pos, rand_val in zip(positions, random_values):
        #     if not random_line_added_to_legend:
        #         ax.plot(
        #             [pos - bar_width / 2, pos + bar_width / 2],
        #             [rand_val, rand_val],
        #             color="black",
        #             linewidth=2,
        #             label="Random vector patch",
        #         )
        #         random_line_added_to_legend = True
        #     else:
        #         ax.plot(
        #             [pos - bar_width / 2, pos + bar_width / 2],
        #             [rand_val, rand_val],
        #             color="black",
        #             linewidth=2,
        #         )

    # Adjust x-axis
    plt.xticks(np.arange(n_groups) + (group_width / 2 - bar_width / 2), proportions.keys())

    plt.title(
        f"Patching singular vectors {pointer_component_dims} extracted from {template_name} task"
    )
    plt.ylabel(f"Proportion of {int(sum(singular_outcomes['box']))} interventions")
    plt.ylim(0, 1)
    plt.xlabel("Intervened template")

    # Add baseline accuracy line
    # plt.axhline(y=0.2, color="r", linestyle="--", label="Random object prediction")

    # Move legend outside and to the upper right
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Intervention behavior")

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    # Adjust subplot parameters to give specified padding
    plt.subplots_adjust(right=0.85)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


# %%
# Intervene on SVD components
reference_layer = 12
intervention_layers = list(range(6, 26))
singular_dims_C = [1, 2]

# Replace targeted singular component with pointer vector (singular vector * mean scale per relation)
act_LbD = einops.rearrange(act_LRBD, "l b r d -> l (b r) d")
_, _, V_DB = torch.svd(act_LbD[reference_layer])
v_DC = V_DB[:, singular_dims_C]
relation_scalar_LRC = (act_LRBD @ v_DC).mean(dim=2)
relation_scalar_LRC_std = (act_LRBD @ v_DC).std(dim=2)

# Random baseline
rand_v_DC = torch.randn_like(v_DC)
rand_v_DC = rand_v_DC / rand_v_DC.norm(dim=0)[None, :] * v_DC.norm(dim=0)[None, :]


# %%
def plot_relation_scalars(
    relation_scalar_LRC,
    relation_scalar_errors_LRC,  # New parameter for errors
    singular_dims_C,
    layers,
    num_relations,
    save_path=None,
    fontsize=16,
):
    """
    Plot relation scalars across layers with error bands in a two-column grid layout.
    
    Args:
        relation_scalar_LRC: Tensor of shape (num_layers, num_relations, num_components)
        relation_scalar_errors_LRC: Tensor of shape (num_layers, num_relations, num_components)
                                  containing standard errors or confidence intervals
        singular_dims_C: List of component indices/dimensions
        layers: List of layer indices
        num_relations: Number of relations/pointers to plot
    """
    # Calculate number of rows needed for two columns
    num_plots = len(singular_dims_C)
    num_rows = (num_plots + 1) // 2  # Ceiling division to handle odd numbers
    
    # Create figure with two columns
    fig, axs = plt.subplots(num_rows, 2, figsize=(16, 5 * num_rows))
    fig.tight_layout(pad=10.0)
    
    # Convert axs to 2D array if there's only one row
    if num_rows == 1:
        axs = np.array([axs])
    
    # Flatten axs for easier iteration
    axs_flat = axs.flatten()
    
    # Define colors for different pointers
    colors = plt.cm.tab10(np.linspace(0, 1, num_relations))
    
    for j_idx, j in enumerate(singular_dims_C):
        ax = axs_flat[j_idx]
        
        # Plot each relation line with error band
        for i in range(num_relations):
            # Extract data and errors for current relation and component
            data = relation_scalar_LRC[:, i, j_idx].numpy()
            errors = relation_scalar_errors_LRC[:, i, j_idx].numpy()
            x = np.arange(len(data))
            
            # Plot error band
            ax.fill_between(
                x,
                data - errors,
                data + errors,
                alpha=0.2,
                color=colors[i],
                label=f"_nolegend_"  # Prevents error bands from appearing in legend
            )
            
            # Plot main line
            ax.plot(
                x, 
                data,
                label=f"Pointer {i}",
                color=colors[i],
                linewidth=2.5
            )
        
        # Increase font sizes
        ax.set_title(f"Projection onto V[{j}]", fontsize=fontsize, pad=15)
        ax.set_xlabel("Layer", fontsize=fontsize, labelpad=10)
        ax.set_ylabel("Projection scalar", fontsize=fontsize, labelpad=10)
        
        # Enhance grid and axis appearance
        ax.grid(True, linestyle='--', alpha=1)
        ax.axhline(0, color="black", linestyle="-", linewidth=1.5)
        ax.set_xlim(0, len(layers) - 1)
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        
        # Enhanced legend
        ax.legend(fontsize=11)
        
        # Add some padding to the y-axis limits
        y_min, y_max = ax.get_ylim()
        padding = (y_max - y_min) * 0.1  # 10% padding
        ax.set_ylim(y_min - padding, y_max + padding)
    
    # Remove any empty subplots if odd number of components
    if num_plots % 2 == 1:
        axs_flat[-1].remove()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.show()


# plot_relation_scalars(
#     relation_scalar_LRC,
#     relation_scalar_LRC_std,
#     singular_dims_C,
#     layers,
#     num_relations,
#     save_path=f"images/relation_scalars_{template_name}_spo{num_samples_per_order_pair}.png",
# )

# %%

v_DC = v_DC.to(model.dtype)
rand_v_DC = rand_v_DC.to(model.dtype)
spo = 50
relation_idxs = range(5)

for rel in relation_idxs:
    print(f"Relation {rel}")

    dataset_paths = {
        "box": f"data/base_source_dataset_box_spo{spo}.json",
        "country": f"data/base_source_dataset_country_spo{spo}.json",
        "month": f"data/base_source_dataset_month_spo{spo}.json",
    }
    base_source_datasets = {k: json.load(open(v, "r")) for k, v in dataset_paths.items()}
    # check all datasets have the same length
    assert len(set([len(v["base"]) for v in base_source_datasets.values()])) == 1
    num_samples = len(base_source_datasets["box"]["base"])

    # # shuffle the datasets
    # for k in base_source_datasets.keys():
    #     indices = np.random.permutation(num_samples)
    #     base_source_datasets[k] = {
    #         "base_dataset": [base_source_datasets[k]["base_dataset"][i] for i in indices],
    #         "source_dataset": [base_source_datasets[k]["source_dataset"][i] for i in indices],
    #     }

    v_outcomes_ds = {
        k: {
            "singular": np.zeros(len(evaluation_labels)),
            "random": np.zeros(len(evaluation_labels)),
        }
        for k in dataset_paths.keys()
    }

    for i in tqdm(range(num_samples), desc="Interventions"):
        for k in base_source_datasets.keys():
            base = base_source_datasets[k]["base"][i]
            source = base_source_datasets[k]["source"][i]
            source_rel = int(source["question"]["answer_key"][1:])
            base_rel = int(base["question"]["answer_key"][1:])
            if source_rel != rel:
                continue

            # Pin singular component interventions

            ## Singular vector
            gen = generate_with_intervention(
                model,
                base,
                layers=intervention_layers,
                intervention_mode="pin",
                pin_direction_DC=v_DC,
                pin_factor_LC=relation_scalar_LRC[:, source_rel],
            )
            out = evaluate_intervention(gen, base, source)
            v_outcomes_ds[k]["singular"] += out

            ## Random vector
            gen = generate_with_intervention(
                model,
                base,
                layers=intervention_layers,
                intervention_mode="pin",
                pin_direction_DC=rand_v_DC,
                pin_factor_LC=relation_scalar_LRC[:, source_rel],
            )
            out = evaluate_intervention(gen, base, source)
            v_outcomes_ds[k]["random"] += out

    # print(v_outcomes_ds)

    plot_generalization(
        v_outcomes_ds,
        evaluation_labels,
        pointer_component_dims=singular_dims_C,
        save_path=f"images/intervention_{rel}_spo{spo}.png",
    )
# %%
noinv_v_outcomes_ds = {
    k: {
        "singular": np.array([0, 100, 0, 0]),
        "random": np.array([0, 100, 0, 0]),
    }
    for k in dataset_paths.keys()
}

plot_generalization(
        noinv_v_outcomes_ds,
        evaluation_labels,
        pointer_component_dims=singular_dims_C,
        save_path=f"images/intervention_{rel}_spo{spo}.png",
    )
# %%
# across full sentences


# prefix_sentence = "To practice my memory, I will list the objects in my attic. "
# sample_sentence = "The key is in Box M, the line is in Box Q, the ball is in Box O, the brush is in Box F, and the mat is in Box H. Box M contains the key."
# suffix_sentence = "I made it! Now let's get some ice cream." 

# total_sentence = prefix_sentence + sample_sentence + suffix_sentence
# encoding = model.tokenizer(total_sentence)
# tokens = model.tokenizer.convert_ids_to_tokens(encoding["input_ids"])


# V_DC with D = 2304 and C = 2
base_sample = base_dataset[0]
base_text = base_sample["text"]
encoding = model.tokenizer(base_text)
tokens = model.tokenizer.convert_ids_to_tokens(encoding["input_ids"])

layers = range(model._model.config.num_hidden_layers)
resid_post_LBPD = []
with torch.no_grad(), model.trace(base_text):
    for layer_idx, layer in enumerate(layers):
        resid_post_module = model.model.layers[layer]
        resid_post_LBPD.append(resid_post_module.output[0].save())
resid_post_LPD = torch.cat(resid_post_LBPD, dim=0)

resid_post_LPD = resid_post_LPD / resid_post_LPD.norm(dim=-1, keepdim=True)

resid_post_LPD = resid_post_LPD.to(torch.float).cpu()
component_LPC = resid_post_LPD @ v_DC.to(torch.float).cpu()
pos_range = torch.arange(resid_post_LPD.shape[1])

#%%
import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_entity_attribute_components(base_sample, component_LPC, tokens, layer_idx):
    # Extract positions from base_sample
    entity_positions = [base_sample['key_to_position'][f'e{i}'] for i in range(5)]
    attribute_positions = [base_sample['key_to_position'][f'a{i}'] for i in range(5)]
    
    # Create the main plot with increased figure size
    plt.figure(figsize=(20, 8))
    plt.title(f"Layer {layer_idx}")
    
    # Plot the component values
    pos_range = np.arange(component_LPC.shape[1])
    plt.plot(pos_range, component_LPC[layer_idx, :, 0], color='gray', alpha=0.5)
    
    # Plot entities and attributes
    plt.scatter(entity_positions, component_LPC[layer_idx, entity_positions, 0], 
               color='blue', s=100, label='Entities', zorder=5)
    plt.scatter(attribute_positions, component_LPC[layer_idx, attribute_positions, 0], 
               color='red', s=100, label='Attributes', zorder=5)
    
    # Connect corresponding entities and attributes with lines
    for i in range(5):
        e_pos = base_sample['key_to_position'][f'e{i}']
        a_pos = base_sample['key_to_position'][f'a{i}']
        plt.plot([e_pos, a_pos], 
                [component_LPC[layer_idx, e_pos, 0], component_LPC[layer_idx, a_pos, 0]], 
                '--', color='green', alpha=0.5)
    
    # Plot all token positions
    # Filter out padding tokens (usually token ID 1)
    valid_positions = [i for i, token in enumerate(tokens) if token != '<pad>']
    plt.xticks(valid_positions, 
               [tokens[i] for i in valid_positions],
               rotation=45, ha='right', fontsize=8)
    
    # Add labels and grid
    plt.xlabel("Position / Token")
    plt.ylabel("Projection Scalar")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.show()

# Function to plot all layers
def plot_all_layers(base_sample, component_LPC, tokens, num_layers):
    for layer_idx in range(num_layers):
        plot_entity_attribute_components(base_sample, component_LPC, tokens, layer_idx)

# Function to plot specific layers
def plot_specific_layers(base_sample, component_LPC, tokens, layers):
    for layer_idx in layers:
        plot_entity_attribute_components(base_sample, component_LPC, tokens, layer_idx)


num_layers = len(component_LPC)  # Number of layers in your model
plot_all_layers(base_sample, component_LPC, tokens, num_layers)
# %%

import matplotlib.pyplot as plt
import numpy as np
import torch

def prepare_activations(model, base_sample, v_DC):
    """Prepare model activations and compute cosine similarities."""
    # Get base text and tokenize
    base_text = base_sample["text"]
    encoding = model.tokenizer(base_text)
    tokens = model.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    
    # Get activations for each layer
    layers = range(model._model.config.num_hidden_layers)
    resid_post_LBPD = []
    
    with torch.no_grad(), model.trace(base_text):
        for layer_idx, layer in enumerate(layers):
            resid_post_module = model.model.layers[layer]
            resid_post_LBPD.append(resid_post_module.output[0].save())
    
    # Concatenate and normalize
    resid_post_LPD = torch.cat(resid_post_LBPD, dim=0)
    resid_post_LPD = resid_post_LPD / resid_post_LPD.norm(dim=-1, keepdim=True)
    resid_post_LPD = resid_post_LPD.to(torch.float).cpu()
    
    return resid_post_LPD, tokens

def plot_2d_cosine_similarities(base_sample, resid_post_LPD, v_DC, tokens, layer_idx):
    """Plot 2D cosine similarities for a specific layer."""
    # Normalize vectors for cosine similarity
    v1 = v_DC[:, 0] / torch.norm(v_DC[:, 0])
    v2 = v_DC[:, 1] / torch.norm(v_DC[:, 1])
    
    # Calculate cosine similarities
    print(resid_post_LPD[layer_idx].shape, v1.shape)
    cos_sim_v1 = resid_post_LPD[layer_idx] @ v1
    cos_sim_v2 = resid_post_LPD[layer_idx] @ v2
    
    # Create figure
    plt.figure(figsize=(10, 10))
    plt.title(f'Layer {layer_idx}: Cosine Similarities with Components')
    
    # Get positions for different token groups
    entity_positions = [base_sample['key_to_position'][f'e{i}'] for i in range(5)]
    attribute_positions = [base_sample['key_to_position'][f'a{i}'] for i in range(5)]
    question_position = [base_sample['key_to_position']['a_question']]
    the_positions = [i for i, token in enumerate(tokens) 
                    if token.lower().strip("▁ ") == 'the']
    box_positions = [i for i, token in enumerate(tokens) 
                    if token.lower().strip("▁ ") == 'box']

    print(f'num the: {len(the_positions)}, num box: {len(box_positions)}')
    
    # Create set of special positions
    special_positions = set(entity_positions + attribute_positions + the_positions + box_positions + question_position)
    
    # Get other positions (excluding padding tokens)
    other_positions = [i for i, token in enumerate(tokens) 
                      if i not in special_positions and token.strip() != '<pad>']
    
    # Plot each group with different colors and markers
    plt.scatter(cos_sim_v1[entity_positions], cos_sim_v2[entity_positions], 
                c='blue', label='Entities', s=100)
    plt.scatter(cos_sim_v1[attribute_positions], cos_sim_v2[attribute_positions], 
                c='red', label='Attributes', s=100)
    plt.scatter(cos_sim_v1[question_position], cos_sim_v2[question_position], 
                c='orange', label='Question', s=100, marker='*')
    plt.scatter(cos_sim_v1[the_positions], cos_sim_v2[the_positions], 
                c='green', label='"the" tokens', s=100)
    plt.scatter(cos_sim_v1[box_positions], cos_sim_v2[box_positions], 
                c='purple', label='"Box" tokens', s=100)
    plt.scatter(cos_sim_v1[other_positions], cos_sim_v2[other_positions], 
                c='gray', alpha=0.5, label='Other tokens', s=50)
    
    # Add token labels for entities, attributes, and question
    for pos in entity_positions + attribute_positions + question_position:
        plt.annotate(tokens[pos], 
                    (cos_sim_v1[pos], cos_sim_v2[pos]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8)
    
    # Add grid and labels
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('Cosine Similarity with v1')
    plt.ylabel('Cosine Similarity with v2')
    plt.legend()
    
    # Make plot square with equal axes
    plt.axis('equal')
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def plot_all_layers_2d(model, base_sample, v_DC):
    """Process data and plot all layers."""
    # Prepare activations
    resid_post_LPD, tokens = prepare_activations(model, base_sample, v_DC)
    
    # Plot each layer
    num_layers = len(resid_post_LPD)
    for layer_idx in range(num_layers):
        plot_2d_cosine_similarities(base_sample, resid_post_LPD, v_DC, tokens, layer_idx)

# Example usage:
plot_all_layers_2d(model, base_sample, v_DC)
# %%


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from IPython.display import HTML, display

def prepare_activations(model, base_sample, v_DC):
    """Prepare model activations and compute cosine similarities."""
    # Get base text and tokenize
    base_text = base_sample["text"]
    encoding = model.tokenizer(base_text)
    tokens = model.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
    
    # Get activations for each layer
    layers = range(model._model.config.num_hidden_layers)
    resid_post_LBPD = []
    
    with torch.no_grad(), model.trace(base_text):
        for layer_idx, layer in enumerate(layers):
            resid_post_module = model.model.layers[layer]
            resid_post_LBPD.append(resid_post_module.output[0].save())
    
    # Concatenate and normalize
    resid_post_LPD = torch.cat(resid_post_LBPD, dim=0)
    resid_post_LPD = resid_post_LPD / resid_post_LPD.norm(dim=-1, keepdim=True)
    resid_post_LPD = resid_post_LPD.to(torch.float).cpu()
    
    return resid_post_LPD, tokens

def plot_interactive_cosine_similarities(model, base_sample, v_DC, save_path=None):
    """Create interactive Plotly plot of cosine similarities across all layers."""
    # Prepare activations
    resid_post_LPD, tokens = prepare_activations(model, base_sample, v_DC)
    resid_post_LPD = resid_post_LPD.to(torch.float)
    v_DC = v_DC.to(torch.float)
    num_layers = len(resid_post_LPD)
    
    # Initialize figure
    fig = go.Figure()
    
    # Normalize direction vectors
    v1 = v_DC[:, 0] / torch.norm(v_DC[:, 0])
    v2 = v_DC[:, 1] / torch.norm(v_DC[:, 1])
    
    # Get positions for different token groups
    entity_positions = [base_sample['key_to_position'][f'e{i}'] for i in range(5)]
    attribute_positions = [base_sample['key_to_position'][f'a{i}'] for i in range(5)]
    question_position = [base_sample['key_to_position']['a_question']]
    the_positions = [i for i, token in enumerate(tokens) 
                    if token.lower().strip("▁ ") == 'the']
    box_positions = [i for i, token in enumerate(tokens) 
                    if token.lower().strip("▁ ") == 'box']
    
    # Create set of special positions
    special_positions = set(entity_positions + attribute_positions + 
                          the_positions + box_positions + question_position)
    
    # Get other positions (excluding padding tokens)
    other_positions = [i for i, token in enumerate(tokens) 
                      if i not in special_positions and token.strip() != '<pad>']
    
    # Create traces for each layer
    for layer_idx in range(num_layers):
        cos_sim_v1 = resid_post_LPD[layer_idx] @ v1
        cos_sim_v2 = resid_post_LPD[layer_idx] @ v2
        
        # Add traces for each token group
        traces = [
            go.Scatter(
                x=cos_sim_v1[entity_positions].numpy(),
                y=cos_sim_v2[entity_positions].numpy(),
                mode='markers+text',
                name='Entities',
                text=[tokens[pos] for pos in entity_positions],
                textposition="top center",
                marker=dict(size=12, color='blue'),
                visible=(layer_idx == 0),
                showlegend=True,  # Always show legend
                legendgroup='entities',
                legendgrouptitle_text='Token types'
            ),
            go.Scatter(
                x=cos_sim_v1[attribute_positions].numpy(),
                y=cos_sim_v2[attribute_positions].numpy(),
                mode='markers+text',
                name='Attributes',
                text=[tokens[pos] for pos in attribute_positions],
                textposition="top center",
                marker=dict(size=12, color='red'),
                visible=(layer_idx == 0),
                showlegend=True,
                legendgroup='attributes'
            ),
            go.Scatter(
                x=cos_sim_v1[question_position].numpy(),
                y=cos_sim_v2[question_position].numpy(),
                mode='markers+text',
                name='Question Box Label',
                text=[tokens[pos] for pos in question_position],
                textposition="top center",
                marker=dict(size=15, color='orange', symbol='star'),
                visible=(layer_idx == 0),
                showlegend=True,
                legendgroup='question'
            ),
            go.Scatter(
                x=cos_sim_v1[the_positions].numpy(),
                y=cos_sim_v2[the_positions].numpy(),
                mode='markers',
                name='"the" tokens',
                marker=dict(size=10, color='green'),
                visible=(layer_idx == 0),
                showlegend=True,
                legendgroup='the'
            ),
            go.Scatter(
                x=cos_sim_v1[box_positions].numpy(),
                y=cos_sim_v2[box_positions].numpy(),
                mode='markers',
                name='"box" tokens',
                marker=dict(size=10, color='purple'),
                visible=(layer_idx == 0),
                showlegend=True,
                legendgroup='box'
            ),
            go.Scatter(
                x=cos_sim_v1[other_positions].numpy(),
                y=cos_sim_v2[other_positions].numpy(),
                mode='markers',
                name='Other tokens',
                marker=dict(size=8, color='gray', opacity=0.5),
                visible=(layer_idx == 0),
                showlegend=True,
                legendgroup='other'
            )
        ]
        
        for trace in traces:
            fig.add_trace(trace)
    
    # Create steps for slider
    steps = []
    num_traces_per_layer = len(traces)
    for layer_idx in range(num_layers):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            label=f"Layer {layer_idx}"
        )
        # Make traces for current layer visible
        for trace_idx in range(num_traces_per_layer):
            step["args"][0]["visible"][layer_idx * num_traces_per_layer + trace_idx] = True
        steps.append(step)
    
    # Calculate global min/max for axis ranges
    x_coords = []
    y_coords = []
    for layer_idx in range(num_layers):
        cos_sim_v1 = resid_post_LPD[layer_idx] @ v1
        cos_sim_v2 = resid_post_LPD[layer_idx] @ v2
        x_coords.extend(cos_sim_v1.numpy())
        y_coords.extend(cos_sim_v2.numpy())
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Add some padding to the ranges (5% on each side)
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    
    # Update layout
    fig.update_layout(
        title="Cosine Similarities of activations with singular vectors<br>Use arrow keys or slider to navigate layers<br>Scroll down to see the full sentence",
        xaxis_title="Cosine Similarity with V[1]",
        yaxis_title="Cosine Similarity with V[2]",
        xaxis=dict(
            range=[x_min - x_padding, x_max + x_padding],
            scaleanchor="y",
            scaleratio=1,
            fixedrange=True  # Prevent zoom/pan
        ),
        yaxis=dict(
            range=[y_min - y_padding, y_max + y_padding],
            fixedrange=True  # Prevent zoom/pan
        ),
        uirevision=True,  # Maintain UI state across updates
        width=800,
        height=800,
        showlegend=True,
        sliders=[dict(
            active=0,
            currentvalue={"prefix": "Layer: "},
            pad={"t": 50},
            steps=steps
        )],
    )

    # Create tokens display div
    filtered_tokens = [token for token in tokens if token.strip() != '<pad>']
    token_spans = []
    for i, token in enumerate(filtered_tokens):
        # Determine token type and color
        color = 'gray'  # default color
        if any(i == pos for pos in entity_positions):
            color = 'blue'
        elif any(i == pos for pos in attribute_positions):
            color = 'red'
        elif any(i == pos for pos in question_position):
            color = 'orange'
        elif any(i == pos for pos in the_positions):
            color = 'green'
        elif any(i == pos for pos in box_positions):
            color = 'purple'
            
        token_spans.append(f'<span style="color: {color}; margin: 0 2px;">{token}</span>')
    
    tokens_div = f"""
    <div id="tokens-display" style="
        margin-top: 20px;
        padding: 15px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 14px;
        line-height: 1.6;
        overflow-x: auto;
        white-space: nowrap;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    ">
        <div style="font-weight: bold; margin-bottom: 8px; color: #495057;">Token Sequence:</div>
        <div style="padding: 8px; background-color: white; border-radius: 4px;">
            {''.join(token_spans)}
        </div>
    </div>
    """

    arrow_key_js = """
    <script>
    (function() {
        // Function to set up keyboard navigation
        function setupKeyboardNavigation() {
            var plotDivs = document.getElementsByClassName('plotly-graph-div');
            if (plotDivs.length === 0) {
                setTimeout(setupKeyboardNavigation, 100);
                return;
            }
            
            var plotDiv = plotDivs[0];
            
            // Handle arrow key events
            function handleArrowKey(event) {
                // Only handle arrow keys if not in an input/textarea
                if (!event.target.tagName.match(/input|textarea/i)) {
                    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
                        event.preventDefault();
                        
                        // Get current slider state
                        var slider = plotDiv._fullLayout.sliders[0];
                        var currentStep = slider.active || 0;
                        var numSteps = slider.steps.length;
                        
                        // Calculate new step
                        var newStep = currentStep + (event.key === 'ArrowRight' ? 1 : -1);
                        newStep = Math.max(0, Math.min(newStep, numSteps - 1));
                        
                        // Update plot if step changed
                        if (newStep !== currentStep) {
                            var update = slider.steps[newStep].args[0].visible;
                            
                            Plotly.update(plotDiv, 
                                {visible: update},
                                {'sliders[0].active': newStep}
                            );
                        }
                    }
                }
            }
            
            // Set up event listeners
            document.removeEventListener('keydown', handleArrowKey);
            document.addEventListener('keydown', handleArrowKey);
        }
        
        // Initialize when document is ready
        if (document.readyState === 'complete') {
            setupKeyboardNavigation();
        } else {
            window.addEventListener('load', setupKeyboardNavigation);
        }
    })();
    </script>
    """

    if save_path is not None:
        # Generate and save HTML
        fig_html = fig.to_html(full_html=True, include_plotlyjs=True)
        fig_html = fig_html.replace("</body>", f"{tokens_div}{arrow_key_js}</body>")
        with open(save_path, "w") as f:
            f.write(fig_html)
    else:
        # Display in notebook
        display(HTML(arrow_key_js))
        fig.show()
    
    return fig



# Example usage:
fig = plot_interactive_cosine_similarities(model, base_sample, v_DC)
# Optional: Save to HTML file
fig = plot_interactive_cosine_similarities(model, base_sample, v_DC, save_path="images/cosine_similarities_v.html")
# %%
base_sample['text']
# %%
