#%%
%load_ext autoreload
%autoreload 2

# %%
import os
import torch
import json
from tqdm import tqdm
import numpy as np
import einops

import pathlib
from utils.project_config import INTERIM_DIR, PLOT_DIR
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
num_samples_per_order_pair = 5  # post filter

random_offset = False
max_context_length = 64
ctx_length_with_pad = 64

layers = range(model._model.config.num_hidden_layers)

template_name = "box"
base_entity_name = "common_objects1"
source_entity_name = "common_objects2"
attribute_name = "letter"

# template_name = "country"
# base_entity_name = "first_names1"
# source_entity_name = "first_names2"
# attribute_name = "letter"

# template_name = "month"
# base_entity_name = "first_names1"
# source_entity_name = "first_names2"
# attribute_name = "letter"

force_recompute_data = False
force_recompute_acts = False

# %%
dataset_path = INTERIM_DIR / f"base_source_dataset_{template_name}_spo{num_samples_per_order_pair}_pad{random_offset}.json"
activation_path = INTERIM_DIR / f"source_activations_{template_name}_spo{num_samples_per_order_pair}_pad{random_offset}.pt"

if force_recompute_data or not os.path.exists(dataset_path):
    dataset_dict = generate_base_source_dataset(
        model,
        num_samples_per_order_pair,
        template_name=template_name,
        base_entity_name=base_entity_name,
        source_entity_name=source_entity_name,
        attribute_name=attribute_name,
        tokenizer=model.tokenizer,
        random_offset=random_offset,
        max_context_length=max_context_length,
        ctx_length_with_pad=ctx_length_with_pad,
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
# Prepare activations and do SVD

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


# %% Compute SVD

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


#%%
# # Full counterfactual patching

from utils.graphing import plot_subspace_patching_multiple_tasks

v_DC = v_DC.to(model.dtype)
rand_v_DC = rand_v_DC.to(model.dtype)

dataset_paths = {
    "box": INTERIM_DIR / f"base_source_dataset_box_spo{num_samples_per_order_pair}_pad{random_offset}.json",
    # "country": INTERIM_DIR / f"base_source_dataset_country_spo{num_samples_per_order_pair}_pad{random_offset}.json",
    # "month": INTERIM_DIR / f"base_source_dataset_month_spo{num_samples_per_order_pair}_pad{random_offset}.json",
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

relation_idxs = range(5)
for rel in relation_idxs:
    print(f"Relation {rel}")

    # v_outcomes_ds = {
    #     k: {
    #         "singular": np.zeros(len(evaluation_labels)),
    #         "random": np.zeros(len(evaluation_labels)),
    #     }
    #     for k in dataset_paths.keys()
    # }

    for i in tqdm(range(num_samples), desc="Interventions"):
        for k in base_source_datasets.keys():
            base = base_source_datasets[k]["base"][i]
            source = base_source_datasets[k]["source"][i]
            source_rel = int(source["question"]["answer_key"][1:])
            base_rel = int(base["question"]["answer_key"][1:])
            if source_rel != rel:
                continue


            ## Pin to mean activation across singular vector
            # gen = generate_with_intervention(
            #     model,
            #     base,
            #     layers=list(range(26)),
            #     intervention_mode="pin",
            #     pin_direction_DC=v_DC,
            #     pin_factor_LC=relation_scalar_LRC[:, source_rel],
            # )
            
            ## Patch full question box token, all resid dimensions

            gen = generate_with_intervention(
                model,
                base,
                layers=[12],
                intervention_mode="full_replace",
                source_act_LD=act_LBED[:, i, -1, :],
            )

            ## Patch full question box token, all resid dimensions
            # gen = generate_with_intervention(
            #     model,
            #     base,
            #     layers=intervention_layers,
            #     intervention_mode="full_replace",
            #     source_act_LD=act_LBED[:, i, -1, :],
            # )

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

plot_subspace_patching_multiple_tasks(
    v_outcomes_ds,
    evaluation_labels,
    title=f"Patching singular vectors {singular_dims_C} extracted from {template_name} task",
    save_path=PLOT_DIR / f"intervention_{rel}_spo{num_samples_per_order_pair}.png",
)


# %%
# # plot_mean_svd_scalar_per_attribute

# from utils.graphing import plot_mean_svd_scalar_per_attribute

# plot_mean_svd_scalar_per_attribute(
#     relation_scalar_LRC,
#     relation_scalar_LRC_std,
#     singular_dims_C,
#     layers,
#     num_relations,
#     save_path= PLOT_DIR / f"relation_scalars_{template_name}_spo{num_samples_per_order_pair}.png",
# )


# %% [markdown]
# # Plot activations in singular value decomposition interactively

# from utils.graphing import plot_singular_vectors_interactive

# num_vs = 5
# save_path =  PLOT_DIR / f"singular_vectors_{num_vs}_{template_name}_spo{num_samples_per_order_pair}.html"

# plot_singular_vectors_interactive(
#     act_LRBD,
#     num_vs,
#     layers,
#     template_name,
#     save_path=save_path,
#     fixed_svd_layer=reference_layer,
# )
    
# %%
# # Plot demo of no intervention across tasks: plot_subspace_patching_multiple_tasks

# noinv_v_outcomes_ds = {
#     k: {
#         "singular": np.array([0, 100, 0, 0]),
#         "random": np.array([0, 100, 0, 0]),
#     }
#     for k in dataset_paths.keys()
# }

# plot_subspace_patching_multiple_tasks(
#         noinv_v_outcomes_ds,
#         evaluation_labels,
#         pointer_component_dims=singular_dims_C,
#         save_path= PLOT_DIR / f"intervention_{rel}_spo{spo}.png",
#     )


# %%
# # Cosine similatities for V[1], V[2] for single tokens in a sentence

# from utils.graphing import plot_svd_cosine_similarities_per_token_over_layers_interactive

# print(f"Using sample: {base_dataset[0]}")

# fig = plot_svd_cosine_similarities_per_token_over_layers_interactive(
#     model, base_dataset[0], v_DC, save_path= PLOT_DIR / "cosine_similarities_v.html"
# )