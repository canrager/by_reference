"""
Cache activations [layer, batch, key, hidden_dim]
"""

import json
from typing import Any, Dict, List, Tuple
import torch
from transformers import BatchEncoding
from nnsight import LanguageModel
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import pickle as pkl

def load_processed_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load a processed dataset from a JSON file.
    """
    with open(dataset_path, "r") as f:
        return json.load(f)

def load_model(name: str):
    """
    Load a pre-trained model.
    """
    if name == "gemma-2-2b":
        model = LanguageModel(
            "google/gemma-2-2b",
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/can/models",
            dispatch=True,
        )
    elif name == "gemma-2-9b":
        model = LanguageModel(
            "google/gemma-2-9b",
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/can/models",
            dispatch=True,
        )
    else:
        raise ValueError(f"Model '{name}' not supported.")

    return model


def get_activation_cache_for_entity(
    model: LanguageModel,
    dataset: List[Dict[str, Any]],
    layers: List[int] = [12],
    n_samples: int = None,
    llm_batch_size: int = 32,
    save_dtype: torch.dtype = torch.float32,
    save_path: str = 'data/activation_cache.pt',
) -> torch.Tensor:
    """
    Compute the activation cache for a specific entity across all samples.
    Each position for an entity instance will be saved as a separate row.

    Args:
        model: The language model
        dataset: List of dataset samples
        layer: Model layer to extract activations from
        n_samples: Number of samples to process (None for all)
        llm_batch_size: Batch size for processing
    """

    if n_samples is not None:
        assert n_samples <= len(
            dataset
        ), "n_samples must be less than or equal to the dataset size."
        dataset = dataset[:n_samples]

    activation_cache = []

    # Create progress bar
    num_batches = (len(dataset) + llm_batch_size - 1) // llm_batch_size
    progress_bar = tqdm(total=num_batches, desc=f"Caching activations")

    for batch_idx in range(0, len(dataset), llm_batch_size):
        torch.cuda.empty_cache()
        batch = dataset[batch_idx : batch_idx + llm_batch_size]

        # Prepare input tensors
        token_ids = torch.tensor([example["token_ids"] for example in batch]).to(model.device)
        attention_masks = torch.tensor([example["attention_masks"] for example in batch]).to(
            model.device
        )

        # Get positions for current batch
        batch_positions = []  # List to store (batch_idx, position) pairs
        for sample_idx, sample in enumerate(batch):
            positions = sample["word_to_position"].values()
            for pos in positions:
                batch_positions.append((sample_idx, pos))
        assert batch_positions, "No positions found in batch"

        # Convert to tensor format
        batch_indices = torch.tensor([bp[0] for bp in batch_positions]).to(model.device)
        position_indices = torch.tensor([bp[1] for bp in batch_positions]).to(model.device)
        batch_encoding = BatchEncoding({"input_ids": token_ids, "attention_mask": attention_masks})

        # Get activations
        batch_cache = torch.zeros(len(layers), len(batch_positions), model.config.hidden_size, device=model.device)
        tracer_kwargs = {"scan": False, "validate": False}
        with torch.no_grad(), model.trace(batch_encoding, **tracer_kwargs):
            for layer in layers:
                resid_post_module = model.model.layers[layer]
                resid_post_BLD = resid_post_module.output[0] # Residual stream activations at idx 0 of model output
                # Extract activations for each position individually
                resid_post_BED = resid_post_BLD[batch_indices, position_indices, :]
                batch_cache[layer] = resid_post_BED.save()

        # Append to activation cache
        activation_cache.append(batch_cache)
        progress_bar.update(1)
    progress_bar.close()
    activation_cache = torch.cat(activation_cache, dim=1)

    if save_path:
        torch.save(activation_cache.cpu().to(save_dtype), save_path)
        print(f"Activation cache saved to {save_path}")

    return activation_cache

def test_activation_cache(
    dataset: List[Dict[str, Any]],
    model: LanguageModel,
    activation_cache: torch.Tensor,
    layers: List[int],
):
    """
    Test the activation cache dimensions and content.

    Args:
        dataset: The input dataset
        model: The language model used
        activation_cache: The computed activation cache with shape [layer, batch, hidden_dim]
        layers: List of layer indices being tested

    Raises:
        AssertionError: If any test fails
    """
    # Test 1: Verify number of layers matches expected
    assert (
        activation_cache.shape[0] == len(layers)
    ), f"Layer dimension mismatch. Expected {len(layers)} layers, got {activation_cache.shape[0]}"

    # Test 2: Verify hidden dimension matches model config
    expected_hidden_dim = model.config.hidden_size
    assert (
        activation_cache.shape[2] == expected_hidden_dim
    ), f"Hidden dimension mismatch. Expected {expected_hidden_dim}, got {activation_cache.shape[2]}"

    # Test 3: Verify number of positions matches total entity positions across dataset
    total_positions = sum(
        len(pos_list) 
        for sample in dataset
        for pos_list in sample["word_to_position"].values()
    )
    assert (
        activation_cache.shape[1] == total_positions
    ), f"Number of positions mismatch. Expected {total_positions} positions, got {activation_cache.shape[1]}"

    # Test 4: Verify no NaN or infinite values
    assert not torch.isnan(
        activation_cache
    ).any(), "Found NaN values in activation cache"
    assert not torch.isinf(
        activation_cache
    ).any(), "Found infinite values in activation cache"

    # Print test results
    print(f"\nTest results for activation cache:")
    print(f"✓ Number of layers: {activation_cache.shape[0]} (matches expected layers)")
    print(f"✓ Number of positions: {activation_cache.shape[1]} (matches dataset)")
    print(f"✓ Hidden dimension: {activation_cache.shape[2]} (matches model config)")
    print(f"✓ No NaN or infinite values detected")
    print(f"✓ Shape of activation cache: {activation_cache.shape}")

    # Additional dataset statistics
    position_counts = [
        len(sample["word_to_position"])
        for sample in dataset
    ]
    min_positions = min(position_counts)
    max_positions = max(position_counts)
    avg_positions = sum(position_counts) / len(position_counts)

    print(f"\nDataset statistics:")
    print(f"- Min positions per sample: {min_positions}")
    print(f"- Max positions per sample: {max_positions}")
    print(f"- Average positions per sample: {avg_positions:.2f}")



if __name__ == "__main__":
    model_name = "gemma-2-2b"
    layers = [25]
    
    print("Loading dataset and model...")
    dataset = load_processed_dataset("data/processed_dataset.json")
    model = load_model(model_name)
    print(f"Loaded dataset with {len(dataset)} samples")

    activation_cache = get_activation_cache_for_entity(
        model, 
        dataset, 
        layers=layers,
        activation_cache_path="data/activation_cache.pt"
    )

    # Run tests
    test_activation_cache(dataset, model, activation_cache, layers)
