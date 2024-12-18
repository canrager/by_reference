import json
from typing import Any, Dict, List
import torch
from transformers import BatchEncoding
from nnsight import LanguageModel
from tqdm.auto import tqdm

def load_processed_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load a processed dataset from a JSON file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)
    
def load_model(name: str):
    """Load a pre-trained model."""
    if name == 'gemma-2-2b':
        model = LanguageModel(
            'google/gemma-2-2b',
            device_map='cuda:0',
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/can/models",
            dispatch=True
        )
    elif name == 'gemma-2-9b':
        model = LanguageModel(
            'google/gemma-2-9b',
            device_map='cuda:0',
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/can/models",
            dispatch=True
        )
    else:
        raise ValueError(f"Model '{name}' not supported.")
    
    return model

def count_entity_positions(dataset: List[Dict[str, Any]], entity_key: str) -> int:
    """Count total number of positions for a given entity across all samples."""
    total_positions = 0
    for sample in dataset:
        positions = sample['entity_attribute_token_positions'][entity_key]
        total_positions += len(positions)
    return total_positions

def get_activation_cache_for_entity(
        model: LanguageModel, 
        dataset: List[Dict[str, Any]],
        entity_key: str,  # Now using "E_0", "E_1" format
        layer: int = 12,
        n_samples: int = None,
        llm_batch_size: int = 8,
    ) -> torch.Tensor:
    """
    Compute the activation cache for a specific entity across all samples.
    Each position for an entity instance will be saved as a separate row.
    """
    if n_samples is not None:
        dataset = dataset[:n_samples]

    activation_cache = []
    num_batches = (len(dataset) + llm_batch_size - 1) // llm_batch_size
    progress_bar = tqdm(total=num_batches, desc=f"Processing {entity_key}")
    
    for batch_idx in range(0, len(dataset), llm_batch_size):
        torch.cuda.empty_cache()
        batch = dataset[batch_idx:batch_idx + llm_batch_size]
        
        # Prepare input tensors
        token_ids = torch.tensor([example['token_ids'] for example in batch]).to(model.device)
        attention_masks = torch.tensor([example['attention_masks'] for example in batch]).to(model.device)
        
        # Get positions for current batch
        batch_positions = []  # List to store (batch_idx, position) pairs
        for sample_idx, sample in enumerate(batch):
            positions = sample['entity_attribute_token_positions'][entity_key]
            for pos in positions:
                batch_positions.append((sample_idx, pos))
        
        if not batch_positions:  # Skip if no positions in batch
            progress_bar.update(1)
            continue
            
        # Convert to tensor format
        batch_indices = torch.tensor([bp[0] for bp in batch_positions]).to(model.device)
        position_indices = torch.tensor([bp[1] for bp in batch_positions]).to(model.device)
        
        batch_encoding = BatchEncoding({
            "input_ids": token_ids,
            "attention_mask": attention_masks
        })

        # Get activations
        tracer_kwargs = {'scan': False, 'validate': False}
        with torch.no_grad(), model.trace(batch_encoding, **tracer_kwargs):
            resid_post_module = model.model.layers[layer]
            resid_post_BLD = resid_post_module.output[0]
            # Extract activations for each position individually
            resid_post_BED = resid_post_BLD[batch_indices, position_indices, :]
            resid_post_BED.save()
            
        activation_cache.append(resid_post_BED)
        progress_bar.update(1)

    progress_bar.close()

    if activation_cache:
        activation_cache = torch.cat(activation_cache, dim=0)
    else:
        raise ValueError(f"No positions found for entity {entity_key} in the dataset")
        
    return activation_cache

def test_activation_cache(dataset: List[Dict[str, Any]], model: LanguageModel, activation_cache: torch.Tensor, entity_key: str):
    """Test the activation cache dimensions and content."""
    expected_hidden_dim = model.config.hidden_size
    assert activation_cache.shape[1] == expected_hidden_dim, \
        f"Hidden dimension mismatch. Expected {expected_hidden_dim}, got {activation_cache.shape[1]}"
    
    expected_rows = count_entity_positions(dataset, entity_key)
    assert activation_cache.shape[0] == expected_rows, \
        f"Number of rows mismatch for {entity_key}. Expected {expected_rows} positions, got {activation_cache.shape[0]}"
    
    assert not torch.isnan(activation_cache).any(), f"Found NaN values in activation cache for {entity_key}"
    assert not torch.isinf(activation_cache).any(), f"Found infinite values in activation cache for {entity_key}"
    
    print(f"\nTest results for {entity_key}:")
    print(f"✓ Hidden dimension: {activation_cache.shape[1]} (matches model config)")
    print(f"✓ Number of positions: {activation_cache.shape[0]} (matches dataset)")
    print(f"✓ No NaN or infinite values detected")
    print(f"✓ Shape of activation cache: {activation_cache.shape}")
    
    position_counts = [len(sample['entity_attribute_token_positions'][entity_key]) for sample in dataset]
    min_positions = min(position_counts)
    max_positions = max(position_counts)
    avg_positions = sum(position_counts) / len(position_counts)
    
    print(f"\nDataset statistics for {entity_key}:")
    print(f"- Min positions per sample: {min_positions}")
    print(f"- Max positions per sample: {max_positions}")
    print(f"- Average positions per sample: {avg_positions:.2f}")

if __name__ == '__main__':
    model_name = 'gemma-2-9b'
    layer = 21

    print("Loading dataset and model...")
    dataset = load_processed_dataset('data/processed_dataset.json')
    model = load_model(model_name)
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Process and test each entity
    for entity_key in ["E_0", "E_1"]:  # Changed to use new entity keys
        activation_cache = get_activation_cache_for_entity(model, dataset, entity_key, layer=layer)
        activation_cache = activation_cache.to(torch.float)
        
        # Run tests before saving
        test_activation_cache(dataset, model, activation_cache, entity_key)
        
        # Save the cache
        save_path = f'artifacts/activation_cache_{entity_key}.pt'
        torch.save(activation_cache.cpu(), save_path)
        print(f"\nSuccessfully saved activation cache to {save_path}")

    print("\nProcessing complete! All activation caches have been saved.")