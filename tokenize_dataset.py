import json
from transformers import AutoTokenizer
import torch
from typing import Dict, List, Tuple, Any
import re


def load_dataset(filename: str = "dataset.json") -> List[Dict[str, Any]]:
    """Load the generated dataset from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def find_entity_attribute_positions(
    text: str, entities: List[str], attributes: Dict[str, str]
) -> Dict[str, Tuple[int, int]]:
    """
    Find the start and end character positions of entities and attributes in the text.
    Returns a dictionary mapping entity/attribute names to their (start, end) positions.
    """
    positions = {}

    # Find positions for entities
    for i, entity in enumerate(entities, 1):
        key = f"e{i}"
        matches = list(re.finditer(re.escape(entity), text))
        for match in matches:
            positions[key] = (match.start(), match.end())

    # Find positions for attributes
    for attr_key, attr_value in attributes.items():
        matches = list(re.finditer(re.escape(attr_value), text))
        for match in matches:
            positions[attr_key] = (match.start(), match.end())

    return positions


def get_token_positions(
    text: str, char_positions: Dict[str, Tuple[int, int]], tokenizer, max_length: int
) -> Tuple[Dict[str, int], List[int]]:
    """
    Convert character positions to token positions and return padded token sequence.
    Returns:
        - Dictionary mapping entity/attribute names to their final token positions
        - List of token ids with padding
    """
    # First get offset mapping (without padding)
    tokens_with_offsets = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    offset_mapping = tokens_with_offsets.offset_mapping

    # Then get padded tokens (separate call)
    tokens_padded = tokenizer(
        text,
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        padding_side="right",
        return_tensors="pt",
    )
    token_ids = tokens_padded["input_ids"][0].tolist()
    attention_masks = tokens_padded["attention_mask"][0].tolist()

    # Find the last token index for each entity/attribute
    token_positions = {}

    for key, (start, end) in char_positions.items():
        # Find the last token that overlaps with the entity/attribute
        for token_idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start < end and token_end > 0:  # Token overlaps with the entity/attribute
                last_token_idx = token_idx
                token_positions[key] = last_token_idx

    return token_positions, token_ids, attention_masks


def process_dataset(
    dataset: List[Dict[str, Any]],
    tokenizer,
    max_length: int = 128,  # Adjust this value based on your needs
) -> List[Dict[str, Any]]:
    """
    Process each example in the dataset to include tokenization information.
    All sequences will be padded to max_length.
    """
    processed_dataset = []

    for example in dataset:
        text = example["text"]
        entities = example["entities"]
        attributes = example["attributes"]

        # Get character positions of entities and attributes
        char_positions = find_entity_attribute_positions(text, entities, attributes)

        # Get token positions and padded token sequence
        token_positions, token_ids, attention_masks = get_token_positions(
            text, char_positions, tokenizer, max_length
        )

        # Verify padding
        assert len(token_ids) == max_length, f"Expected length {max_length}, got {len(token_ids)}"

        # Create processed example
        processed_example = {
            "text": text,
            "entities": entities,
            "attributes": attributes,
            "token_ids": token_ids,
            "attention_masks": attention_masks,
            "entity_attribute_token_positions": token_positions,
            "context_type": example["context_type"],
            "template": example["template"],
        }

        processed_dataset.append(processed_example)

    return processed_dataset


def main():
    """Main function to load model, process dataset, and save results."""
    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset()

    # Process the dataset
    print("Processing dataset...")
    processed_dataset = process_dataset(dataset, tokenizer)

    # Save the processed dataset
    print("Saving processed dataset...")
    with open("processed_dataset.json", "w") as f:
        json.dump(processed_dataset, f, indent=2)

    # Print an example
    print("\nExample processed entry:")
    example = processed_dataset[0]
    print(f"\nOriginal text: {example['text']}")
    print(f"\nTokenized form:")

    # Decode individual tokens for visualization
    tokens = tokenizer.convert_ids_to_tokens(example["token_ids"])
    print("\nTokens with positions:")
    for pos, token in enumerate(tokens):
        if pos in example["entity_attribute_token_positions"].values():
            keys = [k for k, v in example["entity_attribute_token_positions"].items() if v == pos]
            print(f"Position {pos}: {token} <-- {', '.join(keys)}")
        else:
            print(f"Position {pos}: {token}")


if __name__ == "__main__":
    main()
