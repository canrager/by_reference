from dataset_utils import generate_dataset, process_dataset, print_processed_example, load_json
from activation_utils import get_activation_cache_for_entity, test_activation_cache, load_model
from transformers import AutoTokenizer
import re
from typing import List, Dict, Any
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os
import einops
import math


# Dataset generation parameters
DATASET_PARAMS = {
    "num_samples": 300,
    "sample_entities_randomly": True,
    "sample_attributes_randomly": True,
    "selected_template_categories": ["box_templates"],
    "templates_file": "data/templates_box.json",
    "entities_file": "data/entities.json",
    "attributes_file": "data/attributes.json",
    "raw_dataset_path": "data/dataset.json",
}

# Experiment parameters
EXPERIMENT_PARAMS = {
    "model_id": "meta-llama/Llama-2-7b-hf",
    # "model_id": "google/gemma-2-2b",
    "max_context_length": 100,
    "total_length": 100,  # Pre- and append padding tokens to the text filling up to total_length
    "random_offset": False,  # Randomly left_pad
    "processed_dataset_path": "data/processed_dataset.json",
    "force_regenerate_dataset": True,
    "force_reprocess_dataset": True,
    "force_recompute_activations": True,
}


def main():
    # Generate dataset
    if EXPERIMENT_PARAMS["force_regenerate_dataset"] or not os.path.exists(
        DATASET_PARAMS["raw_dataset_path"]
    ):
        print("Generating dataset...")
        dataset = generate_dataset(**DATASET_PARAMS)

        # Print examples
        print("\nGenerated Dataset Examples:")
        for i, example in enumerate(dataset[:3]):
            print(f"\nExample {i+1}:")
            print(f"Context: {example['context_type']}")
            print(f"Text: {example['text']}")
    else:
        print("Loading dataset...")
        dataset = load_json(DATASET_PARAMS["raw_dataset_path"])

    # Process and tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(EXPERIMENT_PARAMS["model_id"])
    tokenizer.pad_token = tokenizer.eos_token
    if EXPERIMENT_PARAMS["force_reprocess_dataset"] or not os.path.exists(
        EXPERIMENT_PARAMS["processed_dataset_path"]
    ):
        print("Processing and tokenizing dataset...")
        dataset = process_dataset(
            dataset,
            tokenizer,
            max_context_length=EXPERIMENT_PARAMS["max_context_length"],
            ctx_length_with_pad=EXPERIMENT_PARAMS["total_length"],
            random_offset=EXPERIMENT_PARAMS["random_offset"],
            save_path=EXPERIMENT_PARAMS["processed_dataset_path"],
        )

        print_processed_example(dataset[0], tokenizer)
    else:
        print("Loading processed dataset...")
        dataset = load_json(EXPERIMENT_PARAMS["processed_dataset_path"])

    #######################

    model = load_model(EXPERIMENT_PARAMS["model_id"])
    layers = list(range(model.config.num_hidden_layers))
    num_activation_samples = None

    if EXPERIMENT_PARAMS["force_recompute_activations"] or not os.path.exists(
        "data/activation_cache.pt"
    ):
        act_LBED, pos_BE = get_activation_cache_for_entity(
            model,
            dataset,
            layers=layers,
            n_samples=num_activation_samples,
            llm_batch_size=32,
            save_dtype=torch.float32,
            save_activations_path="data/activation_cache.pt",
            save_positions_path="data/position_indices.pt",
        )
        test_activation_cache(dataset, model, act_LBED, layers)
    else:
        act_LBED = torch.load("data/activation_cache.pt")
        pos_BE = torch.load("data/position_indices.pt")

    key_regex = r"a(\d+)"
    all_keys = dataset[0]["key_to_word"].keys()
    selected_keys = [key for key in all_keys if re.match(key_regex, key)]
    position_mask = torch.tensor([key in selected_keys for key in all_keys], dtype=torch.bool)
    print(f"Selected {sum(position_mask)} keys: {selected_keys}")
    act_LBED = act_LBED[:, :, position_mask]

    act_LBED = act_LBED.cpu().numpy()
    act_LbD = einops.rearrange(act_LBED, "l b e d -> l (b e) d")

    # def pca_all_layers(act_LbD: np.array, n_components: int = 4):
    #     pca_L = {}
    #     # TODO avoid mismatch between layers and act_LBED
    #     for layer_idx, layer in tqdm(enumerate(layers), desc="Fitting PCA per layer"):
    #         pca_L[layer] = PCA(n_components=n_components)
    #         pca_L[layer].fit(act_LbD[layer_idx])
    #     return pca_L

    def scale_data(data, device='cuda:0'):
        data = torch.from_numpy(data)
        data.to(device)
        mean = torch.mean(data, dim=0)
        std_dev = torch.std(data, dim=0)
        #scaled_data = (data - mean) / std_dev
        scaled_data = (data - mean)
        #scaled_data = data
        return scaled_data, mean, std_dev

    def pca_all_layers(act_LbD: np.array, n_components: int = 4, device="cuda:0"):
        """
        Perform PCA on activation data for all layers using torch's SVD approach.

        Args:
            act_LbD (np.array): Activation data with shape [num_layers, num_samples, feature_dim]
            n_components (int): Number of PCA components to keep
            device (str): Device to perform computations on ('cuda' or 'cpu')

        Returns:
            dict: Dictionary containing PCA results for each layer
        """
        pca_L = {}

        for layer_idx, layer in enumerate(tqdm(range(len(act_LbD)), desc="Fitting PCA per layer")):
            # Get data for current layer
            layer_data = act_LbD[layer_idx]

            # Scale the data
            scaled_data, mean_vec, std_vec = scale_data(layer_data, device)

            # Calculate V matrix (principal components)
            _, _, V = torch.pca_lowrank(scaled_data, q=n_components)

            # Store results in a class-like structure to match sklearn.PCA interface
            class LayerPCA:
                def __init__(self, components, mean, std, n_components):
                    self.components_ = components.cpu().numpy()
                    self.mean_ = mean.cpu().numpy()
                    self.std_ = std.cpu().numpy()
                    self.n_components_ = n_components

                def transform(self, X):
                    X_tensor = torch.from_numpy(X).to(self.components_.device)
                    X_centered = X_tensor - torch.from_numpy(self.mean_)
                    return (
                        torch.matmul(X_centered, torch.from_numpy(self.components_)).cpu().numpy()
                    )

                def inverse_transform(self, X):
                    X_tensor = torch.from_numpy(X).to(self.components_.device)
                    X_proj = torch.matmul(X_tensor, torch.from_numpy(self.components_))
                    return (X_proj + torch.from_numpy(self.mean_)).cpu().numpy()

            # Create PCA object for this layer
            pca_L[layer] = LayerPCA(
                components=V[:, :n_components],
                mean=mean_vec,
                std=std_vec,
                n_components=n_components,
            )

        return pca_L

    def plot_pca_grid(pca_L, act_LbD, layers, key_names, component_pairs):
        """
        Plot PCA results for all layers in a grid layout for each component pair.

        Args:
            pca_dict: Dictionary mapping layer numbers to fitted PCA objects
            act_LbD: Tensor of shape [layers, batch * selected_keys, dimensions]
            layers: List of layer numbers to plot
            key_names: List of key names for legend
            component_pairs: List of tuples containing pairs of PCA components to plot [(1,2), (1,4), etc.]
        """

        # Calculate grid dimensions (as square as possible)
        n_plots = len(layers)
        n_rows = math.ceil(math.sqrt(n_plots))
        n_cols = math.ceil(n_plots / n_rows)

        # Colors for different entities
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#17becf",
            "#bcbd22",
            "#7b3294",
            "#d95f02",
            "#66a61e",
            "#386cb0",
            "#e41a1c",
            "#00897b",
            "#8e44ad",
            "#f39c12",
            "#16a085",
            "#c0392b",
            "#2980b9",
        ]

        # Create separate plots for each component pair
        for pair_idx, (comp1, comp2) in enumerate(component_pairs):
            # Create figure and subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
            if n_plots == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)

            # Flatten axes for easier iteration
            axes_flat = axes.flatten()

            # Plot each layer's PCA results
            for idx, layer in enumerate(layers):
                if idx >= n_plots:
                    break

                ax = axes_flat[idx]

                # Transform data using PCA for this layer
                num_keys = len(key_names)
                act_pca_bD = pca_L[layer].transform(act_LbD[idx])

                act_pca_BED = einops.rearrange(act_pca_bD, "(b e) d -> b e d", e=num_keys)

                # Plot points for each entity
                for key_idx, key_name in enumerate(key_names):
                    ax.scatter(
                        act_pca_BED[:, key_idx, comp1 - 1],  # -1 because components are 1-indexed
                        act_pca_BED[:, key_idx, comp2 - 1],
                        c=colors[key_idx],
                        label=f"Order Nr. {key_name}",
                        alpha=0.6,
                    )

                # Add labels and title
                ax.set_xlabel(f"PC{comp1}")
                ax.set_ylabel(f"PC{comp2}")
                ax.set_title(
                    f"Layer {layer}"
                    #+ "Var explained: {pca_L[layer].explained_variance_ratio_[comp1-1:comp2].sum():.2%}"
                )

                if idx == 0:  # Only add legend to first subplot
                    ax.legend()

            # Remove empty subplots if any
            for idx in range(n_plots, len(axes_flat)):
                fig.delaxes(axes_flat[idx])

            plt.tight_layout()
            plt.savefig(f"pca_plot_pc{comp1}_pc{comp2}.png")
            plt.close()

    # Usage example:
    component_pairs = [(1, 2), (1, 3), (2, 3), (3, 4)]  # Component pairs, 1-indexed
    n_components = np.array(component_pairs).max()  # Maximum component number
    pca_dict = pca_all_layers(act_LbD, n_components=n_components)
    plot_pca_grid(
        pca_dict, act_LbD, layers, key_names=selected_keys, component_pairs=component_pairs
    )


if __name__ == "__main__":
    main()
