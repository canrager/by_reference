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
from utils.generation import is_equal

torch.set_grad_enabled(False)

# %%
model_id = "google/gemma-2-2b"
model = load_model(model_id)
model.tokenizer.pad_token = model.tokenizer.eos_token

# %% [markdown]
# ## Cache activations

# %%
# Example usage
num_samples_per_order_pair = 50  # post filter
random_offset = True
ctx_length_with_pad = 1000
# layers = list(range(model._model.config.num_hidden_layers))
layers = range(model._model.config.num_hidden_layers)

template_name = "box"
base_entity_name = "common_objects1"
source_entity_name = "common_objects2"
attribute_name = "letter"

# template_name = "country"
# base_entity_name = "first_names1"
# source_entity_name = "first_names2"
# attribute_name = "country"

# template_name = "weekday_nodot"
# base_entity_name = "first_names1"
# source_entity_name = "first_names2"
# attribute_name = "weekday"

force_recompute_data = False
force_recompute_acts = False

# %%
dataset_path = f"data/base_source_dataset_{template_name}_spo{num_samples_per_order_pair}.json"
activation_path = f"data/source_activations_{template_name}_spo{num_samples_per_order_pair}.pt"

if force_recompute_data or not os.path.exists(dataset_path):
    base_dataset, source_dataset = generate_base_source_dataset(
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
        both_datasets = json.load(f)
    base_dataset = both_datasets["base_dataset"]
    source_dataset = both_datasets["source_dataset"]

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
# ## Generate with interventions functions


# %%
def retrieve_key_from_generation(base, generation):
    for key, word in base["key_to_word"].items():
        if is_equal(word, generation):
            return key


def generate_with_intervention(
    model,
    base,
    source,
    source_act_LD,
    layers,
    intervention_mode,
    pin_direction_D=None,
    pin_factor=None,
):
    """add a diff vector at selected layers"""

    if (intervention_mode == "pin") and ((pin_direction_D == None) or (pin_factor == None)):
        raise ValueError("pin_direction and pin_factor must be set for pin_direction mode")

    token_ids = base["token_ids"]
    token_ids = token_ids[: -base["num_right_pad"]]
    token_ids = torch.tensor(token_ids).unsqueeze(0).to(model.device)

    base_intervention_pos = base["key_to_position"]["a_question"]

    with model.generate(token_ids, max_new_tokens=1):
        for layer_idx, layer in enumerate(layers):
            resid_BPD = model.model.layers[layer].output[0]
            if intervention_mode == "add":
                resid_BPD[:, base_intervention_pos, :] += source_act_LD[layer_idx]
            elif intervention_mode == "pin":
                current_proj_B1 = resid_BPD[:, base_intervention_pos, :] @ pin_direction_D[:, None]
                resid_BPD[:, base_intervention_pos, :] -= current_proj_B1 * pin_direction_D
                resid_BPD[:, base_intervention_pos, :] += pin_factor * pin_direction_D[None, :]
            elif intervention_mode == "full_replace":
                resid_BPD[:, base_intervention_pos, :] = source_act_LD[layer_idx]
            model.model.layers[layer].output = (resid_BPD,)
        generation = model.generator.output.save()
    generation = model.tokenizer.decode(generation[0, -1])

    return generation


def evaluate_intervention(generation, base, source):
    if is_equal(generation, base["special_str"]["base_object_from_source_box_pointer"]):
        return np.array([1, 0, 0, 0, 0, 0, 0])
    elif is_equal(generation, base["special_str"]["base_object_from_source_qbox_in_base"]):
        return np.array([0, 1, 0, 0, 0, 0, 0])
    elif is_equal(generation, base["special_str"]["source_object"]):
        return np.array([0, 0, 1, 0, 0, 0, 0])
    elif is_equal(generation, base["special_str"]["correct_base_object"]):
        return np.array([0, 0, 0, 1, 0, 0, 0])
    elif any([is_equal(generation, word) for word in base["key_to_word"].values()]):
        return np.array([0, 0, 0, 0, 1, 0, 0])
    elif any([is_equal(generation, word) for word in source["key_to_word"].values()]):
        return np.array([0, 0, 0, 0, 0, 1, 0])
    else:  # total_mismatch
        return np.array([0, 0, 0, 0, 0, 0, 1])


evaluation_labels = [
    "source_pointer",
    "true_source_box",
    "source_object",
    "true_base_object",
    "other_base_object",
    "other_source_object",
    "total_mismatch",
]

# %% [markdown]

# ### Single counterfactual


# %%
# # Do interventions
# outcomes = np.array([0, 0, 0, 0, 0, 0, 0])
# for i, (base, source) in tqdm(enumerate(zip(base_dataset, source_dataset)), total=len(base_dataset), desc="Interventions"):
#     gen = generate_with_intervention(model, base, source, act_LBED[:, i, -1], layers)
#     out = evaluate_intervention(gen, base, source)
#     outcomes += out

# print(outcomes)


# %%
def plot_outcomes(outcomes, labels):
    # Calculate percentages as proportions between 0 and 1
    proportions = outcomes / outcomes.sum()

    fig, ax = plt.subplots(dpi=120)
    bars = ax.bar(labels, proportions)
    plt.xticks(rotation=45)

    # Add percentage labels above each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height*100:.1f}%",  # Convert proportion to percentage for display
            ha="center",
            va="bottom",
        )

    plt.title("")
    plt.ylabel(f"Proportion of {sum(outcomes)} interventions")
    plt.xlabel("Intervention behavior")

    # constant line for baseline accuracy
    plt.axhline(y=0.2, color="r", linestyle="--", label="Random choice of base objects")
    plt.legend()

    plt.tight_layout()
    plt.show()


# plot_outcomes(outcomes, labels)
# plot_outcomes(np.array([0,0,0,1,0,0,0]), labels)

# %% [markdown]
# ### Mean vectors

# %%
# We're doing mulitple layers now
# act_RBD = act_LRBD[0, :, :, :]  # layer 12


# source_mean_LRD = act_LRBD.mean(dim=1)
## save mean vectors
# source_mean_path = 'data/source_mean_activations.pt'
# torch.save(source_mean_LRD, source_mean_path)

# # load mean vectors
# with open(source_mean_path, 'rb') as f:
#     source_mean_LRD = torch.load(f)

# %%
# # Do interventions
# mean_outcomes = np.array([0, 0, 0, 0, 0, 0, 0])
# for i, (base, source) in tqdm(enumerate(zip(base_dataset, source_dataset)), total=len(base_dataset), desc="Interventions"):
#     source_relation_order = source_relation_orders[i]
#     source_mean_LD = source_mean_LRD[:, source_relation_order, :]
#     gen = generate_with_intervention(model, base, source, source_mean_LD, layers)
#     out = evaluate_intervention(gen, base, source)
#     mean_outcomes += out

# print(mean_outcomes)

# %%
# plot_outcomes(mean_outcomes, labels)

# %% [markdown]
# ### Single SVD component


def plot_singular_vectors_interactive(
    act_LRBD, num_vs, layers, template_name, fix_y_range=False, save_path=None
):
    num_relations = act_LRBD.shape[1]
    num_samples_per_relation = act_LRBD.shape[2]

    titles = [f"SVD component n={v}<br>explained var=..." for v in range(num_vs)]
    fig = make_subplots(rows=1, cols=num_vs, subplot_titles=titles)

    colors = px.colors.qualitative.Set1[:num_relations]

    svd_results = []
    y_mins = np.zeros(num_vs)
    y_maxs = np.zeros(num_vs)

    for layer_idx in tqdm(range(len(layers)), desc="Computing SVD"):
        act_BD = einops.rearrange(act_LRBD[layer_idx], "b r d -> (b r) d")
        _, S, V = torch.svd(act_BD)
        explained_var = S / S.sum()
        svd_results.append((S, V, explained_var))

        for v in range(num_vs):
            for i in range(num_relations):
                pointer_components = act_LRBD[layer_idx][i] @ V.T[v]
                y_mins[v] = min(y_mins[v], pointer_components.min().item())
                y_maxs[v] = max(y_maxs[v], pointer_components.max().item())

    for layer_idx, layer in enumerate(layers):
        S, V, explained_var = svd_results[layer_idx]

        for v in range(num_vs):
            fig.layout.annotations[v].update(text=f"n={v}<br>exp var={explained_var[v]:.2f}")

            for i in range(num_relations):
                pointer_components = act_LRBD[layer_idx][i] @ V.T[v]

                fig.add_trace(
                    go.Scatter(
                        x=torch.arange(num_samples_per_relation).numpy(),
                        y=pointer_components.numpy(),
                        mode="markers",
                        name=f"Box {i}",
                        legendgroup=f"Box {i}",
                        marker=dict(color=colors[i]),
                        showlegend=(v == 0),
                        visible=layer_idx == 0,
                    ),
                    row=1,
                    col=v + 1,
                )

            fig.update_xaxes(
                title_text="Samples" if v == num_vs // 2 + 1 else None, row=1, col=v + 1
            )
            fig.update_yaxes(
                title_text="V[n] @ resid_activations" if v == 0 else None, row=1, col=v + 1
            )

    steps = []
    for layer_idx in range(len(layers)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            label=f"Layer {layers[layer_idx]}",
        )
        for v in range(num_vs):
            for i in range(num_relations):
                trace_idx = layer_idx * (num_vs * num_relations) + v * num_relations + i
                step["args"][0]["visible"][trace_idx] = True
        steps.append(step)

    if fix_y_range:
        for v in range(num_vs):
            fig.update_yaxes(range=[y_mins[v], y_maxs[v]], row=1, col=v + 1)

    fig.update_layout(
        sliders=[dict(active=0, currentvalue={"prefix": "Layer: "}, pad={"t": 50}, steps=steps)],
        title=f"Projection of resid onto singular vector n, Task: {template_name}",
        height=500,
        showlegend=True,
    )

    if save_path is not None:
        fig.write_html(save_path)

    return fig


# num_vs = 5
# save_path = f"images/singular_vectors_{num_vs}_{template_name}_spo{num_samples_per_order_pair}.html"

# plot_singular_vectors_interactive(
#     act_LRBD,
#     num_vs,
#     layers,
#     template_name,
#     save_path=save_path,
# )

# %%
# Intervene on SVD components
reference_layers = 11
intervention_layers = range(6, 20)
singular_intervention_dim = 1

# Replace targeted singular component with pointer vector (singular vector * mean scale per relation)
act_LbD = einops.rearrange(act_LRBD, "l b r d -> l (b r) d")
_, _, V_DB = torch.svd(act_LbD[reference_layers])
v_D = V_DB[:, singular_intervention_dim]

## Remove current projection
current_proj_lb = act_LbD[intervention_layers] @ v_D[:, None]
current_proj_lbD = current_proj_lb[:, :, None] * v_D[None, None, :]
act_LbD[intervention_layers] -= current_proj_lbD

## Add target projection
relation_scalar_R = (act_LRBD[reference_layers] @ v_D).mean(dim=1)
target_proj_lbD = relation_scalar_R[:, None, None] * v_D[None, None, :]
act_LbD[intervention_layers] += target_proj_lbD


# %%

for rel in range(5):
    print(f"Relation {rel}")

    dataset_paths = {
        "box": "data/base_source_dataset_box_spo50.json",
        "country": "data/base_source_dataset_country_spo50.json",
        "weekday": "data/base_source_dataset_weekday_spo50.json",
    }
    base_source_datasets = {k: json.load(open(v, "r")) for k, v in dataset_paths.items()}
    # check all datasets have the same length
    assert len(set([len(v["base_dataset"]) for v in base_source_datasets.values()])) == 1
    num_samples = len(base_source_datasets["box"]["base_dataset"])

    # # shuffle the datasets
    # for k in base_source_datasets.keys():
    #     indices = np.random.permutation(num_samples)
    #     base_source_datasets[k] = {
    #         "base_dataset": [base_source_datasets[k]["base_dataset"][i] for i in indices],
    #         "source_dataset": [base_source_datasets[k]["source_dataset"][i] for i in indices],
    #     }

    v_outcomes_ds = {
        k: {
            "singular": np.array([0, 0, 0, 0, 0, 0, 0]),
            "random": np.array([0, 0, 0, 0, 0, 0, 0]),
        }
        for k in dataset_paths.keys()
    }

    cnt = 0
    for i in tqdm(range(num_samples), desc="Interventions"):
        for k in base_source_datasets.keys():
            base = base_source_datasets[k]["base_dataset"][i]
            source = base_source_datasets[k]["source_dataset"][i]
            source_rel = int(source["question"]["answer_key"][1:])
            base_rel = int(base["question"]["answer_key"][1:])

            if source_rel != rel:
                continue
            else:
                cnt += 1


            # # Add diff vector interventions 
            # # Replace targeted singular component with pointer vector (singular vector * mean scale per relation)
            # diff_vec_LD = v_vectors_LRD[:, source_rel, :] - v_vectors_LRD[:, base_rel, :]
            # rand_diff_vec_LD = rand_vectors_LRD[:, source_rel, :] - rand_vectors_LRD[:, base_rel, :]
            # # Singular
            # gen = generate_with_intervention(
            #     model, base, source, diff_vec_LD, layers=[10], intervention_mode='add'
            # )
            # # Random baseline
            # gen = generate_with_intervention(
            #     model, base, source, rand_diff_vec_LD, layers=[10], intervention_mode='add'
            # )

            # Pin singular component interventions

            # Singular vector
            gen = generate_with_intervention(
                model,
                base,
                source,
                source_act_LD=None,
                layers=intervention_layers,
                intervention_mode="pin",
                pin_direction_D=v_D,
                pin_factor=relation_scalar_R[source_rel],
            )

            # Random vector


            out = evaluate_intervention(gen, base, source)
            v_outcomes_ds[k]["singular"] += out

            out = evaluate_intervention(gen, base, source)
            v_outcomes_ds[k]["random"] += out

        if cnt >= 20:
            break

    # print(v_outcomes_ds)

    plot_generalization(v_outcomes_ds, evaluation_labels, selected_labels)
# %%


# %%
pointer_component_dims_N = [1, 2]
V_pointer_DN = V[:, pointer_component_dims_N]

v_factor_RN = (act_RBD @ V_pointer_DN).mean(dim=1, keepdim=True)
print(v_factor_RN)
v_vectors_RD = v_factor_RN @ V_pointer_DN.T
v_vectors_LRD = v_vectors_RD.unsqueeze(0)  # Just single layer

# random baseline of same size
rand_vectors_LRD = torch.randn_like(v_vectors_LRD)
rand_vectors_LRD = (
    rand_vectors_LRD
    / rand_vectors_LRD.norm(dim=-1, keepdim=True)
    * v_vectors_LRD.norm(dim=-1, keepdim=True)
)
# %% [markdown]

# ### Single SVD component same task

# %%
# v_outcomes = np.array([0, 0, 0, 0, 0, 0, 0])
# mismatch_generations = []
# for i, (base, source) in tqdm(enumerate(zip(base_dataset, source_dataset)), total=len(base_dataset), desc="Interventions"):
#     source_vec_LD = v_vectors_LRD[:, source_relation_orders[i], :]
#     base_vec_LD = v_vectors_LRD[:, base_relation_orders[i], :]
#     diff_vec_LD = source_vec_LD - base_vec_LD
#     # source_act = act_LBED[:, i, -1]
#     # source_act = (source_act @ V * V).sum(dim=-1)
#     # source_act = source_act.unsqueeze(0)
#     gen = generate_with_intervention(model, base, source, diff_vec_LD, layers, addition=True)
#     out = evaluate_intervention(gen, base, source)
#     v_outcomes += out

#     if out[-1] == 1:
#         # print('mismatch')
#         # print(f"base {base['text']}")
#         # print(f"source {source['text']}")
#         # print(f"gen {gen}")
#         mismatch_generations.append(gen)

#     if i >= 200:
#         break
# print(f'labels: {labels}')
# print(f'v_outcomes: {v_outcomes}')
# print(f'mismatch_generations: {set(mismatch_generations)}')

# plot_outcomes(v_outcomes, labels)

# %% [markdown]
# ### Single SVD component with mean, task generalization

# %%
selected_labels = [
    "source_pointer",
    "true_base_object",
    "true_source_box",
    "other_base_object",
    "total_mismatch",
]


def plot_generalization(v_outcomes_ds, labels, selected_labels):
    selected_indices = [labels.index(label) for label in selected_labels]

    # Separate singular and random outcomes
    singular_outcomes = {k: v["singular"][selected_indices] for k, v in v_outcomes_ds.items()}
    random_outcomes = {k: v["random"][selected_indices] for k, v in v_outcomes_ds.items()}

    # Normalize other_base_object by 2 # TODO: make this more general
    for k in singular_outcomes.keys():
        singular_outcomes[k][labels.index("other_base_object") - 1] /= 2
        random_outcomes[k][labels.index("other_base_object") - 1] /= 2

    # Calculate percentages
    proportions = {k: v / v.sum() for k, v in singular_outcomes.items()}
    random_proportions = {k: v / v.sum() for k, v in random_outcomes.items()}

    labels = selected_labels

    # Set up the plot
    fig, ax = plt.subplots(dpi=120)

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

        # if label == "other_base_object":
        #     bars = ax.bar(positions, [j/2 for j in values], bar_width, label=label + "_avg")
        #     # Add random baseline lines
        #     for pos, rand_val in zip(positions, random_values):
        #         ax.plot([pos, pos + bar_width], [rand_val/2, rand_val/2],
        #                color='black', linewidth=2)
        # else:

        bars = ax.bar(positions, values, bar_width, label=label)
        # Add random baseline lines
        for pos, rand_val in zip(positions, random_values):
            if not random_line_added_to_legend:
                ax.plot(
                    [pos - bar_width / 2, pos + bar_width / 2],
                    [rand_val, rand_val],
                    color="black",
                    linewidth=2,
                    label="Random vector patch",
                )
                random_line_added_to_legend = True
            else:
                ax.plot(
                    [pos - bar_width / 2, pos + bar_width / 2],
                    [rand_val, rand_val],
                    color="black",
                    linewidth=2,
                    label="",
                )

        # # Add percentage labels above each bar
        # for pos, val in zip(positions, values):
        #     if label == "other_base_object":
        #         ax.text(pos, val/2, f"{val/2*100:.1f}%", ha="center", va="bottom")
        #     else:
        #         ax.text(pos, val, f"{val*100:.1f}%", ha="center", va="bottom")

    # Adjust x-axis
    plt.xticks(np.arange(n_groups) + (group_width / 2 - bar_width / 2), proportions.keys())

    plt.title(f"Patching singular vectors {pointer_component_dims_N} of task {attribute_name}")
    plt.ylabel(f"Proportion of {sum(singular_outcomes['box'])} interventions")
    plt.ylim(0, 1)
    plt.xlabel("Task template")

    # Add baseline accuracy line
    plt.axhline(y=0.2, color="r", linestyle="--", label="Random object prediction")
    plt.legend(loc="upper left", title="Intervention behavior")

    plt.tight_layout()
    plt.show()


# %%

for rel in range(5):
    print(f"Relation {rel}")

    dataset_paths = {
        "box": "data/base_source_dataset_box_spo50.json",
        "country": "data/base_source_dataset_country_spo50.json",
        "weekday": "data/base_source_dataset_weekday_spo50.json",
    }
    base_source_datasets = {k: json.load(open(v, "r")) for k, v in dataset_paths.items()}
    # check all datasets have the same length
    assert len(set([len(v["base_dataset"]) for v in base_source_datasets.values()])) == 1
    num_samples = len(base_source_datasets["box"]["base_dataset"])

    # # shuffle the datasets
    # for k in base_source_datasets.keys():
    #     indices = np.random.permutation(num_samples)
    #     base_source_datasets[k] = {
    #         "base_dataset": [base_source_datasets[k]["base_dataset"][i] for i in indices],
    #         "source_dataset": [base_source_datasets[k]["source_dataset"][i] for i in indices],
    #     }

    v_outcomes_ds = {
        k: {
            "singular": np.array([0, 0, 0, 0, 0, 0, 0]),
            "random": np.array([0, 0, 0, 0, 0, 0, 0]),
        }
        for k in dataset_paths.keys()
    }

    cnt = 0
    for i in tqdm(range(num_samples), desc="Interventions"):
        for k in base_source_datasets.keys():
            base = base_source_datasets[k]["base_dataset"][i]
            source = base_source_datasets[k]["source_dataset"][i]
            source_rel = int(source["question"]["answer_key"][1:])
            base_rel = int(base["question"]["answer_key"][1:])

            if source_rel != rel:
                continue
            else:
                cnt += 1

            diff_vec_LD = v_vectors_LRD[:, source_rel, :] - v_vectors_LRD[:, base_rel, :]
            rand_diff_vec_LD = rand_vectors_LRD[:, source_rel, :] - rand_vectors_LRD[:, base_rel, :]

            # Singular
            gen = generate_with_intervention(
                model, base, source, diff_vec_LD, layers=[10], addition=True
            )
            out = evaluate_intervention(gen, base, source)
            v_outcomes_ds[k]["singular"] += out

            # Random baseline
            gen = generate_with_intervention(
                model, base, source, rand_diff_vec_LD, layers=[10], addition=True
            )
            out = evaluate_intervention(gen, base, source)
            v_outcomes_ds[k]["random"] += out

        if cnt >= 20:
            break

    # print(v_outcomes_ds)

    plot_generalization(v_outcomes_ds, evaluation_labels, selected_labels)
# %%
