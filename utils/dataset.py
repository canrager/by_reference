import re
import json
import random
from typing import Dict, List, Tuple, Any, Optional
import os
from collections import defaultdict
from transformers import AutoTokenizer
from itertools import product
from utils.generation import gen_and_eval
from nnsight import LanguageModel
from tqdm import tqdm


def load_json(filename: str) -> Dict:
    """Load a JSON file and return its contents."""
    with open(filename, "r") as f:
        return json.load(f)


def get_words(
    word_dict: Dict[str, List[str]],
    num_words: int,
    prefix: str,
    sample_randomly: bool = True,
    selected_categories: Optional[List[str]] = None,
    exclude_values: List[str] = [],
) -> Dict[str, str]:
    """
    Select n unique words from dataset, optionally from specific categories.

    Raises:
        ValueError: If there aren't enough unique entities available.
    """
    all_words = []
    if selected_categories is None:
        for words in word_dict.values():
            all_words.extend(words)
    else:
        for category in selected_categories:
            if category not in word_dict:
                raise ValueError(f"Category '{category}' not found in word_dict.")
            all_words.extend(word_dict[category])

    unique_words = list(set(all_words))
    unique_words = [word for word in unique_words if word not in exclude_values]

    if len(unique_words) < num_words:
        raise ValueError(
            f"Not enough unique words available. Required: {num_words}, "
            f"Available: {len(unique_words)} Unique words: {unique_words}"
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


def format_example(
    e_dict: Dict[str, str], a_dict: Dict[str, str], template: str, question_order: int
) -> Dict[str, Any]:
    combined_dict = e_dict | a_dict
    question_key = f"a{question_order}"
    answer_key = f"e{question_order}"
    question_dict = {
        "question_key": question_key,
        "question_str": a_dict[question_key],
        "answer_key": answer_key,
        "answer_str": e_dict[answer_key],
    }
    example = {
        "text": template.format(**combined_dict),
        "entity_keys": list(e_dict.keys()),
        "attribute_keys": list(a_dict.keys()),
        "key_to_word": combined_dict,
        "question": question_dict,
    }
    return example


def generate_dataset(
    num_samples: int,
    selected_entity_categories: Optional[List[str]] = None,
    selected_attribute_categories: Optional[List[str]] = None,
    sample_entities_randomly: bool = True,
    sample_attributes_randomly: bool = True,
    selected_template_categories: Optional[List[str]] = None,
    question_order: Optional[int] = None,
    entities_file: str = "data/entities.json",
    attributes_file: str = "data/attributes.json",
    templates_file: str = "data/templates_box.json",
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

        e_dict = get_words(
            entities_data,
            num_relations,
            "e",
            sample_attributes_randomly,
            selected_entity_categories,
        )
        a_dict = get_words(attributes_data, num_relations, "a", sample_entities_randomly, selected_attribute_categories)

        if question_order is None:
            question_order = random.choice(range(num_relations))
        a_dict["a_question"] = a_dict[f"a{question_order}"]

        dataset.append(format_example(e_dict, a_dict, template, question_order))

    if not dataset:
        raise ValueError("Could not generate any valid samples")

    if raw_dataset_path:
        with open(raw_dataset_path, "w") as f:
            json.dump(dataset, f, indent=2)

    return dataset


def generate_base_source_pair(
    source_order: int,
    base_order: int,
    template: str,
    num_relations: int,
    box_labels: List[str],
    entity_data: Dict[str, List[str]],
    base_entity_name: str,
    source_entity_name: str,
    attribute_name: str,
    verbose: bool = False,
):

    # Select objects
    source_object_dict = get_words(
        entity_data, num_relations, "e", selected_categories=[source_entity_name]
    )
    base_object_dict = get_words(
        entity_data, num_relations, "e", selected_categories=[base_entity_name]
    )

    # generate source box labels
    source_box_dict = get_words(box_labels, num_relations, "a", selected_categories=[attribute_name])
    source_box_dict["a_question"] = source_box_dict[f"a{source_order}"]

    # generate base box labels
    base_box_dict = get_words(
        box_labels,
        num_relations,
        "a",
        selected_categories=[attribute_name],
        exclude_values=source_box_dict.values(),
    )
    base_box_dict["a_question"] = base_box_dict[f"a{base_order}"]
    # randomly assign the source question label to one of neiter base_order or source_order
    source_box_in_base_order = random.choice(
        [i for i in range(num_relations) if i not in [base_order, source_order]]
    )
    base_box_dict[f"a{source_box_in_base_order}"] = source_box_dict["a_question"]

    # Format examples
    source_example = format_example(source_object_dict, source_box_dict, template, source_order)
    base_example = format_example(base_object_dict, base_box_dict, template, base_order)

    # Add data for evaluation
    base_example["special_str"] = {
        "base_object_from_source_box_pointer": base_example['key_to_word'][f"e{source_order}"],
        "base_object_from_source_qbox_in_base": base_example['key_to_word'][f"e{source_box_in_base_order}"],
        "correct_base_object": base_example['question']["answer_str"],
        "source_object": source_example['key_to_word'][f"e{source_order}"],
    }

    return source_example, base_example


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
        add_special_tokens=True,
    )
    input_ids = tokens_with_offsets["input_ids"]
    content_length = len(input_ids)

    if content_length > max_context_length:
        raise AssertionError(
            f"Context length {content_length} exceeds maximum allowed length of {max_context_length}"
        )

    if random_offset:
        max_offset = total_length - max_context_length
        num_left_pad = random.randint(0, max_offset)
    else:
        num_left_pad = 0
    num_right_pad = total_length - num_left_pad - content_length

    pad_token_id = tokenizer.pad_token_id
    token_ids = [pad_token_id] * num_left_pad + input_ids + [pad_token_id] * num_right_pad
    attention_masks = (
        [0] * num_left_pad
        + [1] * content_length
        + [0] * (total_length - num_left_pad - content_length)
    )

    offset_mapping = tokens_with_offsets["offset_mapping"]
    tokens = [text[s:e].strip().lower() for s, e in offset_mapping]
    token_to_position = defaultdict(list)
    for pos, token in enumerate(tokens):
        token_to_position[token].append(pos)

    key_positions = {}
    for key, word in key_to_word.items():
        word_lower = word.lower().strip(" ")
        if word_lower not in token_to_position or not token_to_position[word_lower]:
            raise AttributeError(f"Unable to match {key}={word} with available tokens {token_to_position}.")
        key_positions[key] = token_to_position[word_lower].pop(0) + num_left_pad

    return key_positions, token_ids, attention_masks, num_left_pad, content_length, num_right_pad


def print_processed_example(processed_example: Dict[str, Any], tokenizer):
    """Print a processed example with token position mapping."""
    print(f"Original text: {processed_example['text']}")
    print(f"key_to_word: {processed_example['key_to_word']}")
    print(f'key_to_position: {processed_example["key_to_position"]}')
    print(f"\nTokenized form:")

    tokens = tokenizer.convert_ids_to_tokens(processed_example["token_ids"])
    initial_padding = sum(1 for token in tokens if token == tokenizer.pad_token)
    print(f"Number of padding tokens (not shown): {initial_padding}")

    position_to_word = {v: k for k, v in processed_example["key_to_position"].items()}
    for pos, token in enumerate(tokens):
        if token == tokenizer.pad_token:
            continue
        token = token.replace("Ġ", "_")
        if pos in position_to_word:
            print(f"Position {pos}: {token} <-- {position_to_word[pos]}")
        else:
            print(f"Position {pos}: {token}")


def process_example(
    example: Dict[str, Any],
    tokenizer,
    max_context_length: int = 100,
    ctx_length_with_pad: int = 11000,
    random_offset: bool = True,
) -> Dict[str, Any]:
    """Process example with token position mapping."""
    text = example["text"]
    key_to_word = example["key_to_word"]

    (
        token_positions,
        token_ids,
        attention_masks,
        num_left_pad,
        ctx_length_no_pad,
        num_right_pad,
    ) = get_token_positions(
        text=text,
        key_to_word=key_to_word,
        tokenizer=tokenizer,
        max_context_length=max_context_length,
        total_length=ctx_length_with_pad,
        random_offset=random_offset,
    )

    question = example["question"]
    question["answer_token_str"] = key_to_word[question["answer_key"]]
    question["answer_token_pos"] = token_positions[question["answer_key"]]
    question["answer_token_id"] = token_ids[question["answer_token_pos"]]

    processed_example = {
        "key_to_position": token_positions,
        "num_left_pad": num_left_pad,
        "ctx_length_no_pad": ctx_length_no_pad,
        "num_right_pad": num_right_pad,
        "ctx_length_with_pad": ctx_length_with_pad,
        "total_ctx_length": ctx_length_with_pad,
        "token_ids": token_ids,
        "attention_masks": attention_masks,
        "question": question,
    }
    for key, value in example.items():
        if key not in processed_example:
            processed_example[key] = value

    return processed_example


def process_dataset(
    dataset: List[Dict[str, Any]],
    tokenizer,
    max_context_length: int = 100,
    ctx_length_with_pad: int = 1000,
    random_offset: bool = True,
    save_path: Optional[str] = "data/processed_dataset.json",
) -> List[Dict[str, Any]]:
    """Process dataset with token position mapping."""
    processed_dataset = []
    for example in dataset:
        processed_example = process_example(
            example,
            tokenizer,
            max_context_length=max_context_length,
            ctx_length_with_pad=ctx_length_with_pad,
            random_offset=random_offset,
        )
        processed_dataset.append(processed_example)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(processed_dataset, f, indent=2)

    return processed_dataset


def generate_base_source_dataset(
    model: LanguageModel,
    num_samples_per_order_pair: int,
    template_name: str,
    base_entity_name: str,
    source_entity_name: str,
    attribute_name: str,
    tokenizer: AutoTokenizer,
    max_context_length: int = 100,
    ctx_length_with_pad: int = 1100,
    random_offset: bool = True,
    save_path: Optional[str] = None,
):
    # load template
    template = load_json("data/templates_box.json")[template_name][0]
    num_relations = count_unique_relations(template)

    # load entities
    entity_data = load_json("data/entities.json")

    # load attributes / box labels
    box_labels = load_json("data/attributes.json")

    # Iterate over allowed source and base orders
    full_base_dataset = []
    full_source_dataset = []
    for source_order, base_order in tqdm(
        product(range(num_relations), repeat=2),
        desc="Generating dataset",
        total=num_relations ** 2,
    ):
        if source_order == base_order:
            continue
        pair_base_dataset = []
        pair_source_dataset = []
        while len(pair_base_dataset) < num_samples_per_order_pair:
            source_example, base_example = generate_base_source_pair(
                source_order,
                base_order,
                template,
                num_relations,
                box_labels,
                entity_data,
                base_entity_name=base_entity_name,
                source_entity_name=source_entity_name,
                attribute_name=attribute_name,
            )
            source_example = process_example(source_example, tokenizer, max_context_length, ctx_length_with_pad, random_offset)
            base_example = process_example(base_example, tokenizer, max_context_length, ctx_length_with_pad, random_offset)
            source_is_correct = gen_and_eval(model, source_example)
            base_is_correct = gen_and_eval(model, base_example)
            if source_is_correct and base_is_correct:
                pair_source_dataset.append(source_example)
                pair_base_dataset.append(base_example)

        full_base_dataset.extend(pair_base_dataset)
        full_source_dataset.extend(pair_source_dataset)

    dataset_dict = {
        "base": full_base_dataset,
        "source": full_source_dataset
    }
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(dataset_dict, f, indent=2)
            
    return dataset_dict


if __name__ == "__main__":
    from utils.activation import load_model
    # Example usage
    num_samples = 100
    model_id = "google/gemma-2-2b"
    total_length = 200
    random_offset = True

    
    words = get_words(
        word_dict=load_json("data/attributes.json"),
        num_words=5,
        prefix="a",
        selected_categories=["country"],
        verbose=True,
    )


    # # generate base source pair

    # pair = generate_base_source_pair(
    #     source_order=0,
    #     base_order=1,
    #     template=load_json("data/templates_box.json")["country"][0],
    #     num_relations=5,
    #     box_labels=load_json("data/attributes.json"),
    #     entity_data=load_json("data/entities.json"),
    #     base_entity_name="first_names1",
    #     source_entity_name="first_names2",
    #     attribute_name="country",
    #     verbose=True,
    # )
    # print(pair)


    # model = load_model(model_id)
    # # Generate base source dataset

    # datasets = generate_base_source_dataset(
    #     model=model,
    #     num_samples_per_order_pair=1,
    #     template_name="country",
    #     base_entity_name="first_names1",
    #     source_entity_name="first_names2",
    #     attribute_name="country",
    #     tokenizer=model.tokenizer,
    #     max_context_length=100,
    #     ctx_length_with_pad=1100,
    #     random_offset=True,
    #     save_path="data/processed_base_source_dataset.json",
    # )

    # # Print example
    # print("\nExample from base dataset:")
    # for i in range(10):
    #     print_processed_example(datasets["base"][i], model.tokenizer)

    # Old

    # # Generate dataset
    # dataset = generate_dataset(
    #     num_samples=num_samples,
    #     sample_entities_randomly=True,
    #     sample_attributes_randomly=True,
    #     selected_template_categories=["box_templates"],
    #     templates_file="data/templates_box.json",
    # )

    # # Process and tokenize dataset
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # tokenizer.pad_token = tokenizer.eos_token
    # processed_dataset = process_dataset(
    #     dataset,
    #     tokenizer,
    #     ctx_length_with_pad=total_length,
    #     random_offset=random_offset,
    #     save_path="data/processed_dataset.json",
    # )

    # # Print example
    # print("\nExample from processed dataset:")
    # print_processed_example(processed_dataset[0], tokenizer)
