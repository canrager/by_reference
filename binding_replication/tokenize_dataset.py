import json
from transformers import AutoTokenizer
import torch
from typing import Dict, List, Tuple, Any
import re
import random


def load_dataset(filename: str = "dataset.json") -> List[Dict[str, Any]]:
    """Load the generated dataset from a JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def find_all_entity_positions(text: str, entity: str) -> List[Tuple[int, int]]:
    """Find all occurrences of an entity in the text."""
    return [(m.start(), m.end()) for m in re.finditer(re.escape(entity), text)]


def get_token_positions(
    text: str,
    char_positions: Dict[str, List[Tuple[int, int]]],
    tokenizer,
    max_context_length: int = 100,
    total_length: int = 1000,
    random_offset: bool = True,
) -> Tuple[Dict[str, List[int]], List[int], List[int]]:
    """
    Convert character positions to token positions and add random padding offset.
    """
    # Tokenize without padding first
    base_tokens = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    content_length = len(base_tokens["input_ids"][0])

    # Verify context length
    assert content_length <= max_context_length, \
        f"Context length {content_length} exceeds maximum allowed length of {max_context_length}"

    # Get offset mapping
    tokens_with_offsets = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offset_mapping = tokens_with_offsets.offset_mapping

    # Calculate random offset
    if random_offset:
        max_offset = total_length - max_context_length
        random_offset = random.randint(0, max_offset)
    else:
        random_offset = 0

    # Create padded sequence
    pad_token_id = tokenizer.pad_token_id
    token_ids = (
        ([pad_token_id] * random_offset)
        + base_tokens["input_ids"][0].tolist()
        + ([pad_token_id] * (total_length - random_offset - content_length))
    )

    # Create attention mask
    attention_masks = (
        ([0] * random_offset)
        + ([1] * content_length)
        + ([0] * (total_length - random_offset - content_length))
    )

    # Find token positions for entities and attributes
    token_positions = {}
    for key, positions_list in char_positions.items():
        token_positions[key] = []
        for start, end in positions_list:
            occurrence_tokens = []
            for token_idx, (token_start, token_end) in enumerate(offset_mapping):
                if token_start < end and token_end > start:
                    occurrence_tokens.append(token_idx)
            if occurrence_tokens:
                adjusted_idx = occurrence_tokens[-1] + random_offset
                token_positions[key].append(adjusted_idx)

    return token_positions, token_ids, attention_masks


def find_entity_attribute_positions(
    text: str, 
    entities: List[str], 
    attributes: Dict[str, Dict[str, List[str]]]
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Find positions of entities and attributes in the text.
    Adapted for new dataset format.
    """
    positions = {}

    # Find positions for E_0 and E_1
    for i, entity in enumerate(entities):
        key = f"E_{i}"
        pattern = r"\b" + re.escape(entity) + r"\b"
        positions[key] = [(m.start(), m.end()) for m in re.finditer(pattern, text)]

    # Find positions for A_0 and A_1
    countries = attributes["countries"]
    for i, country in enumerate(countries):
        key = f"A_{i}"
        pattern = r"\b" + re.escape(country) + r"\b"
        positions[key] = [(m.start(), m.end()) for m in re.finditer(pattern, text)]

    # Add qn_subject positions if present in text
    if "qn_subject" in text:
        pattern = r"\b" + re.escape(text["qn_subject"]) + r"\b"
        positions["qn_subject"] = [(m.start(), m.end()) for m in re.finditer(pattern, text)]

    return positions


def process_dataset(
    dataset: List[Dict[str, Any]],
    tokenizer,
    max_context_length: int = 100,
    total_length: int = 1000,
    random_offset: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process dataset to include tokenization information.
    Adapted for new dataset format.
    """
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
                "qn_subject": example["qn_subject"],
                "answer": example["answer"]
            }

            processed_dataset.append(processed_example)

        except AssertionError as e:
            print(f"Skipping example {i} due to length violation: {e}")
            skipped_count += 1
            continue

    if skipped_count > 0:
        print(f"\nWarning: Skipped {skipped_count} examples due to context length exceeding {max_context_length} tokens")

    return processed_dataset


def main():
    """Main function to process and save the dataset."""
    random_offset = True
    total_length = 1000
    model_id = "google/gemma-2b-it"  # Using Gemma model

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset()

    # Process dataset
    print("Processing dataset...")
    processed_dataset = process_dataset(
        dataset, 
        tokenizer, 
        total_length=total_length, 
        random_offset=random_offset
    )

    # Save processed dataset
    print("Saving processed dataset...")
    with open("processed_dataset.json", "w") as f:
        json.dump(processed_dataset, f, indent=2)

    # Print example
    if processed_dataset:
        print("\nExample processed entry:")
        example = processed_dataset[0]
        print(f"\nOriginal text: {example['text']}")
        print(f"\nTokenized form:")

        tokens = tokenizer.convert_ids_to_tokens(example["token_ids"])
        print("\nTokens with positions:")
        
        initial_padding = sum(1 for token in tokens if token == tokenizer.pad_token)
        print(f"\nInitial padding tokens: {initial_padding}")

        for pos, token in enumerate(tokens):
            entity_names = []
            for key, positions in example["entity_attribute_token_positions"].items():
                if pos in positions:
                    entity_names.append(key)

            if entity_names:
                print(f"Position {pos}: {token} <-- {', '.join(entity_names)}")
            else:
                print(f"Position {pos}: {token}")


if __name__ == "__main__":
    main()