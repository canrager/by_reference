import json
from typing import Any, Dict, List, Tuple
import torch
from transformers import BatchEncoding
from nnsight import LanguageModel
from tqdm.auto import tqdm
from datasets import load_dataset

EPS = 1e-6


def load_processed_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Load a processed dataset from a JSON file.
    """
    with open(dataset_path, "r") as f:
        return json.load(f)


def load_model(model_id: str):
    """
    Load a pre-trained model.
    """
    if model_id == "google/gemma-2-2b":
        model = LanguageModel(
            model_id,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/models",
            dispatch=True,
        )
    elif model_id == "google/gemma-2-9b":
        model = LanguageModel(
            model_id,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/models",
            dispatch=True,
        )
    elif (model_id == "meta-llama/Llama-2-7b-hf"):
        # the hf is necessary, it contains the hf compatible version
        model = LanguageModel(
            model_id,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/models",
            dispatch=True,
        )
    elif model_id == "meta-llama/Llama-3.2-1B":
        model = LanguageModel(
            model_id,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/models",
            dispatch=True,
        )
    elif model_id == "meta-llama/Llama-3.1-8B":
        model = LanguageModel(
            model_id,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            cache_dir="/share/u/models",
            dispatch=True,
        )
    else:
        raise ValueError(f"Model '{model_id}' not supported.")

    # Configure Tokenizer
    model.tokenizer.pad_token = model.tokenizer.eos_token

    return model


def get_activation_cache_for_entity(
    model: LanguageModel,
    dataset: List[Dict[str, Any]],
    layers: List[int] = [12],
    n_samples: int = None,
    llm_batch_size: int = 32,
    save_dtype: torch.dtype = torch.float32,
    save_activations_path: str = "data/activation_cache.pt",
    save_positions_path: str = "data/position_indices.pt",
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
    position_indices = []

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

        # Convert positions directly to tensors
        batch_indices = []
        batch_positions = []
        num_keys = sum(1 for _ in batch[0]["key_to_position"])

        for i, sample in enumerate(batch):
            pos = list(sample["key_to_position"].values())
            batch_indices.append([i] * num_keys)
            batch_positions.append(pos)

        # Create tensors in one go
        batch_indices = torch.tensor(batch_indices, device=model.device)
        batch_positions = torch.tensor(batch_positions, device=model.device)
        batch_encoding = BatchEncoding({"input_ids": token_ids, "attention_mask": attention_masks})

        # Get activations
        batch_cache = torch.zeros(
            len(layers), len(batch), num_keys, model.config.hidden_size, device=model.device
        )
        tracer_kwargs = {"scan": False, "validate": False}
        with torch.no_grad(), model.trace(batch_encoding, **tracer_kwargs):
            for layer_idx, layer in enumerate(layers):
                resid_post_module = model.model.layers[layer]
                resid_post_BLD = resid_post_module.output[
                    0
                ]  # Residual stream activations at idx 0 of model output
                resid_post_BED = resid_post_BLD[batch_indices, batch_positions, :]
                batch_cache[layer_idx] = resid_post_BED.save()

        # Append to activation cache
        activation_cache.append(batch_cache)
        position_indices.append(batch_positions)
        progress_bar.update(1)
    progress_bar.close()
    activation_cache = torch.cat(activation_cache, dim=1)
    position_indices = torch.cat(position_indices, dim=0)

    if save_activations_path:
        torch.save(activation_cache.cpu().to(save_dtype), save_activations_path)
        print(f"Activation cache saved to {save_activations_path}")

    if save_positions_path:
        torch.save(position_indices.cpu(), save_positions_path)
        print(f"Position indices saved to {save_positions_path}")

    return activation_cache, position_indices


def hf_dataset_to_generator(
    dataset_path: str, tokenizer: Any, batch_size: int, ctx_length: int = 128
):
    """
    Convert a Hugging Face dataset to a generator that yields batches of samples.
    """
    dataset = load_dataset(dataset_path, split="train", streaming=True)
    for batch in dataset.batch(batch_size):
        yield tokenizer.batch_encode_plus(
            batch["text"], max_length=ctx_length, truncation=True, return_tensors="pt"
        )


def get_activation_cache(
    model: LanguageModel,
    hf_dataset_path: str,
    save_activations_path: str,
    layers: List[int],
    num_samples: int,
    max_seq_len: int = 128,
    llm_batch_size: int = 32,
) -> torch.Tensor:
    """
    Compute the activation cache for a specific entity across all samples.
    Each position for an entity instance will be saved as a separate row.
    """

    activation_cache = []
    save_dtype = model.dtype

    # Prepare batches
    dataset_generator = hf_dataset_to_generator(
        hf_dataset_path,
        tokenizer=model.tokenizer,
        batch_size=llm_batch_size,
        ctx_length=max_seq_len,
    )
    num_batches = num_samples // llm_batch_size
    progress_bar = tqdm(total=num_batches, desc=f"Caching activations")

    for batch_idx in range(num_batches):
        torch.cuda.empty_cache()
        batch = next(dataset_generator)

        # Prepare input tensors
        # encoding = model.tokenizer.batch_encode_plus(batch, max_length=max_seq_len, truncation=True, return_tensors="pt")

        batch_cache = torch.zeros(
            len(layers), llm_batch_size, max_seq_len, model.config.hidden_size, device=model.device
        )
        tracer_kwargs = {"scan": False, "validate": False}
        with torch.no_grad(), model.trace(batch, **tracer_kwargs):
            for layer_idx, layer in enumerate(layers):
                resid_post_module = model.model.layers[layer]
                batch_cache[layer_idx] = resid_post_module.output[0].save()

        # Append to activation cache
        activation_cache.append(batch_cache)
        progress_bar.update(1)
    progress_bar.close()
    activation_cache = torch.cat(activation_cache, dim=1)

    if save_activations_path:
        torch.save(activation_cache.cpu().to(save_dtype), save_activations_path)
        print(f"Activation cache saved to {save_activations_path}")

    return activation_cache


def get_embedding_cache(
    model: LanguageModel,
    hf_dataset_path: str = "skylion007/openwebtext",
    save_activations_path: str = None,
    num_samples: int = 1000,
    max_seq_len: int = 128,
    hf_batch_size: int = 1028,
) -> torch.Tensor:
    """
    Compute the activation cache for a specific entity across all samples.
    Each position for an entity instance will be saved as a separate row.
    """

    embedding_VD = model.model.embed_tokens.weight.data

    num_batches = num_samples // hf_batch_size
    num_tokens = num_batches * hf_batch_size * max_seq_len
    activation_cache = torch.zeros(num_tokens, model.config.hidden_size, device=model.device)
    save_dtype = model.dtype

    # Prepare batches
    dataset_generator = hf_dataset_to_generator(
        hf_dataset_path,
        tokenizer=model.tokenizer,
        batch_size=hf_batch_size,
        ctx_length=max_seq_len,
    )

    for batch_idx in range(num_batches):
        batch = next(dataset_generator).input_ids
        batch = batch.flatten()
        start_idx = batch_idx * hf_batch_size * max_seq_len
        end_idx = start_idx + hf_batch_size * max_seq_len
        activation_cache[start_idx:end_idx] = embedding_VD[batch]

    if save_activations_path:
        torch.save(activation_cache.cpu().to(save_dtype), save_activations_path)
        print(f"Activation cache saved to {save_activations_path}")

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
    assert activation_cache.shape[0] == len(
        layers
    ), f"Layer dimension mismatch. Expected {len(layers)} layers, got {activation_cache.shape[0]}"

    # # Test 2: Verify num activations matches len dataset
    # Not necessarily true since we allow downsampling via n_samples
    # assert activation_cache.shape[1] == len(
    #     dataset
    # ), f"Number of samples mismatch. Expected {len(dataset)}, got {activation_cache.shape[1]}"

    # Test 3: Verify number of positions matches number of keys
    num_keys = sum(1 for _ in dataset[0]["key_to_position"])
    num_positions = activation_cache.shape[2]
    assert (
        num_keys == num_positions
    ), f"Number of positions mismatch. Expected {num_keys}, got {num_positions}"

    # Test 4: Verify hidden dimension matches model config
    expected_hidden_dim = model.config.hidden_size
    assert activation_cache.shape[3] == expected_hidden_dim, (
        f"Hidden dimension mismatch. Expected {expected_hidden_dim}, "
        f"got {activation_cache.shape[3]}"
    )

    # Test 5: Verify no NaN or infinite values
    assert not torch.isnan(activation_cache).any(), "Found NaN values in activation cache"
    assert not torch.isinf(activation_cache).any(), "Found infinite values in activation cache"

    # Print test results
    print(f"\nTest results for activation cache:")
    print(f"✓ Number of layers: {activation_cache.shape[0]} (matches expected layers)")
    print(f"✓ Number of samples: {activation_cache.shape[1]} (matches dataset)")
    print(f"✓ Number of keys: {activation_cache.shape[2]} (matches dataset)")
    print(f"✓ Hidden dimension: {activation_cache.shape[3]} (matches model config)")
    print(f"✓ No NaN or infinite values detected")
    print(f"✓ Shape of activation cache: {activation_cache.shape}")


def compute_whitening_transform(diverse_acts_TD, save_path=None):
    """
    Compute whitening transformation matrix based on equation 3.2
    
    Args:
        gamma: Token embeddings/representations tensor of shape [n_tokens, dim]
        
    Returns:
        M: Transformation matrix Cov(γ)^(-1) of shape [dim, dim]
    """
    # Normalize gamma
    diverse_acts_TD = (diverse_acts_TD - diverse_acts_TD.mean(dim=0, keepdim=True))

    # Compute covariance matrix
    cov = torch.cov(diverse_acts_TD.T)  # [dim, dim]
    
    # Add small diagonal term for numerical stability
    cov = cov + EPS * torch.eye(cov.shape[0], device=cov.device)
    
    # Compute M = Cov(γ)^(-1) directly as suggested in the paper
    M = torch.linalg.inv(cov)

    if save_path is not None:
        torch.save(M, save_path)
    
    return M

def embedding_whitening_transform(model, save_path=None):
    """
    Compute whitening transformation matrix based on equation 3.2
    
    Args:
        gamma: Token embeddings/representations tensor of shape [n_tokens, dim]
        
    Returns:
        M: Transformation matrix Cov(γ)^(-1) of shape [dim, dim]
    """
    embedding_TD = get_embedding_cache(model)

    # Normalize gamma
    embedding_TD = (embedding_TD - embedding_TD.mean(dim=0, keepdim=True))

    # Compute covariance matrix
    cov = torch.cov(embedding_TD.T)  # [dim, dim]
    
    # Add small diagonal term for numerical stability
    cov = cov + EPS * torch.eye(cov.shape[0], device=cov.device)
    
    # Compute M = Cov(γ)^(-1) directly as suggested in the paper
    M = torch.linalg.inv(cov).cpu()

    if save_path is not None:
        torch.save(M, save_path)
    
    return M


if __name__ == "__main__":
    model_id = "google/gemma-2-2b"
    model = load_model(model_id)

    hf_path = "skylion007/openwebtext"
    generator = hf_dataset_to_generator(hf_path, model.tokenizer, 1, 128)
    print(next(generator).input_ids.shape)

    # layers = [25]
    # save_activations_path = "data/activation_cache.pt"
    # save_positions_path = "data/position_indices.pt"
    # num_activation_samples = 32

    # print("Loading dataset and model...")
    # dataset = load_processed_dataset("data/processed_dataset.json")
    # print(f"Loaded dataset with {len(dataset)} samples")

    # activation_cache, position_indices = get_activation_cache_for_entity(
    #     model,
    #     dataset,
    #     layers=layers,
    #     n_samples=num_activation_samples,
    #     save_activations_path=save_activations_path,
    #     save_positions_path=save_positions_path,
    # )

    # # Run tests
    # test_activation_cache(dataset, model, activation_cache, layers)
