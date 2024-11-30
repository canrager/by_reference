import json
from typing import Any, Dict, List
import torch
from transformers import BatchEncoding
from nnsight import LanguageModel

def load_processed_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load a processed dataset from a JSON file.
    """
    with open(dataset_path, 'r') as f:
        return json.load(f)
    
def load_model(name: str):
    """
    Load a pre-trained model.
    """
    if name == 'gemma2b':
        model = LanguageModel(
            'google/gemma-2-2b',
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

def get_activation_cache(
        model: LanguageModel, 
        dataset: List[Dict[str, Any]],
        layer: int = 12,
        n_samples: int = None,
        llm_batch_size: int = 8,

    ) -> Dict[str, torch.Tensor]:
    """
    Compute the activation cache for the given dataset.
    """
    
    if n_samples is not None:
        assert n_samples <= len(dataset), "n_samples must be less than or equal to the dataset size."
        dataset = dataset[:n_samples]

    num_entities_per_sample = len(dataset[0]['entities'])
    hidden_dim = model.config.hidden_size
    
    activation_cache = []
    for batch_idx in range(0, len(dataset), llm_batch_size):
        batch = dataset[batch_idx:batch_idx + llm_batch_size]
        token_ids = [example['token_ids'] for example in batch]
        attention_masks = [example['attention_masks'] for example in batch]
        entity_positions = [
            [example['entity_attribute_token_positions'][f'e{i}'] for i in range(1, num_entities_per_sample + 1)]
            for example in batch
        ]
        token_ids = torch.tensor(token_ids).to(model.device)
        attention_masks = torch.tensor(attention_masks).to(model.device)
        entity_positions = torch.tensor(entity_positions).to(model.device)
        batch_arange = torch.arange(len(batch))
        batch_arange = batch_arange.unsqueeze(1).expand(-1, 3).to(model.device)
        batch_encoding = BatchEncoding({
            "input_ids": token_ids,
            "attention_mask": attention_masks
        })

        tracer_kwargs = {'scan': False, 'validate': False}
        with torch.no_grad(), model.trace(batch_encoding, **tracer_kwargs):
            resid_post_module = model.model.layers[layer]
            resid_post_BLD = resid_post_module.output[0]
            resid_post_BED = resid_post_BLD[batch_arange, entity_positions, :]
            resid_post_BED.save()
        activation_cache.append(resid_post_BED)

    activation_cache = torch.cat(activation_cache, dim=0)
    return activation_cache

if __name__ == '__main__':
    dataset = load_processed_dataset('processed_dataset.json')
    model = load_model('gemma2b')
    activation_cache = get_activation_cache(model, dataset)
    torch.save(activation_cache, 'activation_cache.pt')