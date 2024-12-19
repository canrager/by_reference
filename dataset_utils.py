import re
import json
import random
from typing import Dict, List, Tuple, Any, Optional
import os
from collections import defaultdict
from transformers import AutoTokenizer


def load_json(filename: str) -> Dict:
    """Load a JSON file and return its contents."""
    with open(filename, "r") as f:
        return json.load(f)


def get_words(
    word_dict: Dict[str, List[str]],
    num_words: int,
    prefix: str,
    sample_randomly: bool = True,
) -> Dict[str, str]:
    """
    Select n unique words from dataset, optionally from specific categories.

    Raises:
        ValueError: If there aren't enough unique entities available.
    """
    all_words = []
    for category, words_list in word_dict.items():
        all_words.extend(words_list)

    unique_words = list(set(all_words))

    if len(unique_words) < num_words:
        raise ValueError(
            f"Not enough unique words available. Required: {num_words}, "
            f"Available: {len(unique_words)}"
        )

    if sample_randomly:
        words_list = random.sample(unique_words, num_words)
    else:
        words_list = unique_words[:num_words]

    return {f"{prefix}{i}": word for i, word in enumerate(words_list)}


def count_unique_relations(template: str) -> int:
    """Count the number of unique relations in a given template."""
    relation_pattern = re.compile(r"\{(e\d+)\}")
    matches = relation_pattern.findall(template)
    unique_relations = set(matches)
    return len(unique_relations)


def generate_dataset(
    num_samples: int,
    sample_entities_randomly: bool = True,
    sample_attributes_randomly: bool = True,
    selected_template_categories: Optional[List[str]] = None,
    entities_file: str = "data/entities.json",
    attributes_file: str = "data/attributes.json",
    templates_file: str = "data/templates.json",
    raw_dataset_path: Optional[str] = "data/dataset.json",
) -> List[Dict[str, Any]]:
    """Generate a dataset of formatted templates with unique entities and attributes."""
    entities_data = load_json(entities_file)
    attributes_data = load_json(attributes_file)
    templates_data = load_json(templates_file)

    dataset = []

    for i in range(num_samples):
        if selected_template_categories is None:
            selected_template_categories = list(templates_data.keys())
        context = random.choice(selected_template_categories)
        template = random.choice(templates_data[context])

        num_relations = count_unique_relations(template)

        e_dict = get_words(entities_data, num_relations, "e", sample_attributes_randomly)
        a_dict = get_words(attributes_data, num_relations, "a", sample_entities_randomly)

        q_order = random.choice(range(num_relations))
        q_key = f'a{q_order}'
        q_str = a_dict[q_key]
        q_dict = {'a-question': q_str}

        combined_dict = e_dict | a_dict | q_dict
        formatted_text = template.format(**combined_dict)

        example = {
            "context_type": context,
            "template": template,
            "key_to_word": combined_dict,
            "text": formatted_text,
            "question": {
                "q_key": q_key
            }
        }

        dataset.append(example)

    if not dataset:
        raise ValueError("Could not generate any valid samples")
    
    if raw_dataset_path:
        with open(raw_dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)

    return dataset


def get_token_positions(
    text: str,
    key_to_word: Dict[str, str],
    tokenizer,
    max_context_length: int = 100,
    total_length: int = 1000,
    random_offset: bool = True,
) -> Tuple[Dict[str, List[List[int]]], List[int], List[int]]:
    """Convert character positions to token positions with random padding."""
    tokens_with_offsets = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = tokens_with_offsets["input_ids"]
    content_length = len(input_ids)

    if content_length > max_context_length:
        raise AssertionError(
            f"Context length {content_length} exceeds maximum allowed length of {max_context_length}"
        )

    if random_offset:
        max_offset = total_length - max_context_length
        random_offset_value = random.randint(0, max_offset)
    else:
        random_offset_value = 0

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

    offset_mapping = tokens_with_offsets["offset_mapping"]
    tokens = [text[s:e].strip().lower() for s,e in offset_mapping]
    token_to_position = defaultdict(list)
    for pos, token in enumerate(tokens):
        token_to_position[token].append(pos)

    key_positions = {}
    for key, word in key_to_word.items():
        word_lower = word.lower()
        if word_lower not in token_to_position or not token_to_position[word_lower]:
            raise AttributeError(f'Unable to match {key}={word} with available tokens.')
        key_positions[key] = token_to_position[word_lower].pop(0)

    return key_positions, token_ids, attention_masks


def print_processed_example(processed_example: Dict[str, Any], tokenizer):
    """Print a processed example with token position mapping."""
    print(f"Original text: {processed_example['text']}")
    print(f"key_to_word: {processed_example['key_to_word']}")
    print(f'word_to_position: {processed_example["word_to_position"]}')
    print(f"\nTokenized form:")

    tokens = tokenizer.convert_ids_to_tokens(processed_example["token_ids"])
    initial_padding = sum(1 for token in tokens if token == tokenizer.pad_token)
    print(f"Number of padding tokens (not shown): {initial_padding}")

    position_to_word = {v: k for k, v in processed_example["word_to_position"].items()}
    for pos, token in enumerate(tokens):
        if token == tokenizer.pad_token:
            continue
        if pos in position_to_word:
            print(f"Position {pos}: {token} <-- {position_to_word[pos]}")
        else:
            print(f"Position {pos}: {token}")


def process_dataset(
    dataset: List[Dict[str, Any]],
    tokenizer,
    max_context_length: int = 100,
    total_length: int = 1000,
    random_offset: bool = True,
    save_path: Optional[str] = "data/processed_dataset.json",
) -> List[Dict[str, Any]]:
    """Process dataset with token position mapping."""
    processed_dataset = []
    for example in dataset:
        text = example["text"]
        key_to_word = example["key_to_word"]

        token_positions, token_ids, attention_masks = get_token_positions(
            text=text,
            key_to_word=key_to_word,
            tokenizer=tokenizer,
            max_context_length=max_context_length,
            total_length=total_length,
            random_offset=random_offset           
        )

        processed_example = {
            "context_type": example["context_type"],
            "template": example["template"],
            "text": text,
            "key_to_word": key_to_word,
            "word_to_position": token_positions,
            "token_ids": token_ids,
            "attention_masks": attention_masks,
        }

        processed_dataset.append(processed_example)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(processed_dataset, f, indent=2)

    return processed_dataset


if __name__ == "__main__":
    # Example usage
    num_samples = 5
    model_id = "google/gemma-2-2b"
    total_length = 200
    random_offset = False
    
    # Generate dataset
    dataset = generate_dataset(
        num_samples=num_samples,
        sample_entities_randomly=True,
        sample_attributes_randomly=True,
        selected_template_categories=["box_templates"],
        templates_file="data/templates_box.json",
    )
    
    # Process and tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processed_dataset = process_dataset(
        dataset,
        tokenizer,
        total_length=total_length,
        random_offset=random_offset,
        save_path="data/processed_dataset.json"
    )
    
    # Print example
    print("\nExample from processed dataset:")
    print_processed_example(processed_dataset[0], tokenizer)