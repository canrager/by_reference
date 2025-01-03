import json
from transformers import AutoTokenizer
import torch
from typing import Dict, List, Tuple, Any
import re
import random


def load_dataset(filename: str = "data/dataset.json") -> List[Dict[str, Any]]:
    """Load the generated dataset from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def find_entity_attribute_positions(
    text: str, entities: Dict[str, str], attributes: Dict[str, str]
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Find all positions of entities (box letters) and their corresponding objects in the text.
    Returns character-level positions for both entities and their attributes.
    """
    positions = {}

    # Find positions for box letters (entities)
    for entity_key, entity_value in entities.items():
        # Use word boundary for precise matching of box letters
        pattern = r"\b" + re.escape(entity_value) + r"\b"
        positions[entity_key] = [(m.start(), m.end()) for m in re.finditer(pattern, text)]

    # Find positions for objects (attributes)
    for attr_key, attr_value in attributes.items():
        # Use word boundary for precise matching of objects
        pattern = r"\b" + re.escape(attr_value) + r"\b"
        positions[attr_key] = [(m.start(), m.end()) for m in re.finditer(pattern, text)]

    return positions


def get_token_positions(
    text: str,
    char_positions: Dict[str, List[Tuple[int, int]]],
    tokenizer,
    max_context_length: int = 100,
    total_length: int = 1000,
    random_offset: bool = True,
) -> Tuple[Dict[str, List[List[int]]], List[int], List[int]]:
    """
    Convert character positions to token positions with random padding.
    Now properly handles multi-token spans and maintains all token positions.
    """
    # Tokenize the text and get offset mapping
    tokens_with_offsets = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = tokens_with_offsets["input_ids"][0].tolist()
    offset_mapping = tokens_with_offsets["offset_mapping"][0]

    content_length = len(input_ids)

    # Verify context length
    if content_length > max_context_length:
        raise AssertionError(
            f"Context length {content_length} exceeds maximum allowed length of {max_context_length}"
        )

    # Calculate padding offset
    if random_offset:
        max_offset = total_length - max_context_length
        random_offset_value = random.randint(0, max_offset)
    else:
        random_offset_value = 0

    # Create padded sequences
    pad_token_id = tokenizer.pad_token_id
    token_ids = (
        [pad_token_id] * random_offset_value
        + input_ids
        + [pad_token_id] * (total_length - random_offset_value - content_length)
    )
    attention_masks = (
        [0] * random_offset_value
        + [1] * content_length
        + [0] * (total_length - random_offset_value - content_length)
    )

    # Map character positions to token positions
    token_positions = {}
    for key, positions_list in char_positions.items():
        token_positions[key] = []
        for start, end in positions_list:
            # Find all tokens that overlap with this character span
            token_span = []
            for idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start < end and token_end > start:
                    adjusted_idx = idx + random_offset_value
                    token_span.append(adjusted_idx)

            assert token_span, f"No token spans found for {key}"
            token_positions[key].append(token_span)

    # Extract final token positions
    for key in token_positions:
        token_positions[key] = [positions[-1] for positions in token_positions[key]]

    return token_positions, token_ids, attention_masks


def process_dataset(
    dataset: List[Dict[str, Any]],
    tokenizer,
    max_context_length: int = 100,
    total_length: int = 1000,
    random_offset: bool = True,
) -> List[Dict[str, Any]]:
    """Process dataset with improved token position mapping."""
    processed_dataset = []
    skipped_count = 0

    for i, example in enumerate(dataset):
        text = example["text"]
        entities = example["entities"]
        attributes = example["attributes"]

        try:
            # Get character positions
            char_positions = find_entity_attribute_positions(text, entities, attributes)

            # Get token positions and padded sequence
            token_positions, token_ids, attention_masks = get_token_positions(
                text,
                char_positions,
                tokenizer,
                max_context_length,
                total_length,
                random_offset,
            )

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

        except AssertionError as e:
            print(f"Skipping example {i} due to length violation: {e}")
            skipped_count += 1
            continue

    if skipped_count > 0:
        print(
            f"\nWarning: Skipped {skipped_count} examples due to length exceeding {max_context_length} tokens"
        )

    return processed_dataset


def main():
    random_offset = False
    total_length = 200
    model_id = "google/gemma-2-9b"

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset()

    # Process dataset
    print("Processing dataset...")
    processed_dataset = process_dataset(
        dataset, tokenizer, total_length=total_length, random_offset=random_offset
    )

    # Save processed dataset
    print("Saving processed dataset...")
    with open("data/processed_dataset.json", "w") as f:
        json.dump(processed_dataset, f, indent=2)

    # Print example
    if processed_dataset:
        print("\nExample processed entry:")
        example = processed_dataset[0]
        print(f"Original text: {example['text']}")
        print(f"Entities: {example['entities']}")
        print(f"Attributes: {example['attributes']}")
        print(f'entity attribute positions: {example["entity_attribute_token_positions"]}')
        print(f"\nTokenized form:")

        tokens = tokenizer.convert_ids_to_tokens(example["token_ids"])
        initial_padding = sum(1 for token in tokens if token == tokenizer.pad_token)
        print(f"Number of padding tokens (not shown): {initial_padding}")

        # Improved token position display
        positions_map = {}
        for key, positions in example["entity_attribute_token_positions"].items():
            for pos in positions:
                if pos not in positions_map:
                    positions_map[pos] = []
                positions_map[pos].append(key)

        for pos, token in enumerate(tokens):
            if token == tokenizer.pad_token:
                continue
            if pos in positions_map:
                print(f"Position {pos}: {token} <-- {', '.join(positions_map[pos])}")
            else:
                print(f"Position {pos}: {token}")


if __name__ == "__main__":
    main()
