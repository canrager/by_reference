import re
import json
import random
from typing import Dict, List, Any, Optional
import os


def load_json(filename: str) -> Dict:
    """Load a JSON file and return its contents."""
    with open(filename, "r") as f:
        return json.load(f)


def get_entities(
    entities: Dict[str, List[str]],
    selected_entity_categories: List[str],
    num_entities: int,
    sample_randomly: bool = True,
) -> Dict[str, str]:
    """
    Select n unique entities, optionally from specific categories.

    Raises:
        ValueError: If there aren't enough unique entities available.
    """
    all_entities = []
    for category, entity_list in entities.items():
        if category in selected_entity_categories:
            all_entities.extend(entity_list)

    # Remove duplicates from all_entities
    unique_entities = list(set(all_entities))

    if len(unique_entities) < num_entities:
        raise ValueError(
            f"Not enough unique entities available. Required: {num_entities}, "
            f"Available: {len(unique_entities)}"
        )

    if sample_randomly:
        entity_list = random.sample(unique_entities, num_entities)
    else:
        entity_list = unique_entities[:num_entities]

    return {f"e{i}": entity for i, entity in enumerate(entity_list)}


def get_required_attributes(template: str) -> Dict[str, int]:
    """
    Count how many unique values are needed for each attribute type in the template.
    Returns a dictionary mapping attribute types to required counts.
    """
    attr_counts = {}
    placeholders = re.findall(r"\{([^}]*_[^}]*)\}", template)

    for placeholder in placeholders:
        _, attr_type = placeholder.split("_")
        attr_counts[attr_type] = attr_counts.get(attr_type, 0) + 1

    return attr_counts


def get_attributes(
    template: str,
    attributes: Dict[str, Dict[str, List[str]]],
    sample_randomly: bool = True,
) -> Dict[str, str]:
    """
    Generate unique attribute values for placeholders in a template string.

    Raises:
        ValueError: If there aren't enough unique attributes available or if an
                   attribute type is not found in the attributes dictionary.
    """
    placeholder_to_value = {}
    placeholders = re.findall(r"\{([^}]*_[^}]*)\}", template)

    # Track used values for each attribute type
    used_values = {}
    # Get required counts for each attribute type
    required_counts = get_required_attributes(template)

    # Verify sufficient unique values are available for each attribute type
    for attr_type, count in required_counts.items():
        attr_found = False
        for category, subcategories in attributes.items():
            if attr_type in subcategories:
                attr_found = True
                available_values = list(set(subcategories[attr_type]))  # Remove any duplicates
                if len(available_values) < count:
                    raise ValueError(
                        f"Not enough unique values for attribute '{attr_type}'. "
                        f"Required: {count}, Available: {len(available_values)}"
                    )
                break

        if not attr_found:
            raise ValueError(f"Attribute type '{attr_type}' not found in attributes.")

    # Assign values to placeholders
    for placeholder in placeholders:
        entity, attr_type = placeholder.split("_")

        # Find the category containing this attribute type
        for category, subcategories in attributes.items():
            if attr_type in subcategories:
                values = list(set(subcategories[attr_type]))  # Remove any duplicates

                # Initialize used values tracking for this attribute type
                if attr_type not in used_values:
                    used_values[attr_type] = set()

                # Get available values (those not yet used)
                available_values = [v for v in values if v not in used_values[attr_type]]

                if not available_values:
                    raise ValueError(f"No more unique values available for attribute '{attr_type}'")

                # Select a value
                selected_value = (
                    random.choice(available_values) if sample_randomly else available_values[0]
                )
                used_values[attr_type].add(selected_value)
                placeholder_to_value[placeholder] = selected_value
                break

    return placeholder_to_value


def count_unique_entities(template: str) -> int:
    """
    Count the number of unique entities in a given template.
    Entities are placeholders like {e1}, {e2}, {e3}, ignoring attributes like {e1_zone}.
    """
    entity_pattern = re.compile(r"\{(e\d+)\}")
    matches = entity_pattern.findall(template)
    unique_entities = set(matches)
    return len(unique_entities)


def generate_dataset(
    num_samples: int,
    selected_entity_categories: Optional[List[str]] = None,
    sample_entities_randomly: bool = True,
    sample_attributes_randomly: bool = True,
    selected_template_categories: Optional[List[str]] = None,
    entities_file: str = "data/entities.json",
    attributes_file: str = "data/attributes.json",
    templates_file: str = "data/templates.json",
) -> List[Dict[str, Any]]:
    """
    Generate a dataset of formatted templates with unique entities and attributes.

    Raises:
        ValueError: If there aren't enough unique entities or attributes available,
                   or if an attribute type is not found.
    """
    # Load the JSON files
    entities_data = load_json(entities_file)
    attributes_data = load_json(attributes_file)
    templates_data = load_json(templates_file)

    dataset = []

    for i in range(num_samples):
        try:
            # Select random context and template
            if selected_template_categories is None:
                selected_template_categories = list(templates_data.keys())
            context = random.choice(selected_template_categories)
            template = random.choice(templates_data[context])

            # Find number of entities required in the template
            num_entities = count_unique_entities(template)

            # Get unique entities
            entity_dict = get_entities(
                entities_data, selected_entity_categories, num_entities, sample_entities_randomly
            )

            # Get unique attributes based on template
            attr_dict = get_attributes(template, attributes_data, sample_attributes_randomly)

            # Format the template
            combined_dict = entity_dict | attr_dict
            print(f"combined_dict: {combined_dict}")
            print(f"template: {template}")
            formatted_text = template.format(**combined_dict)

            # Store the example
            example = {
                "context_type": context,
                "template": template,
                "entities": entity_dict,
                "attributes": attr_dict,
                "text": formatted_text,
            }

            dataset.append(example)

        except ValueError as e:
            print(f"Error generating sample {i + 1}: {str(e)}")
            continue

    if not dataset:
        raise ValueError("Could not generate any valid samples due to insufficient unique values")

    return dataset


def save_dataset(dataset: List[Dict[str, Any]], output_file: str = "data/dataset.json"):
    """Save the generated dataset to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)


if __name__ == "__main__":
    num_samples = 100
    selected_entity_categories = ["capital_letters"]
    selected_template_categories = ["nonbox_templates"]
    templates_file = "data/templates_box.json"
    # NOTE the template defines the required attribute_categories and num_entities
    sample_enitites_randomly = True
    sample_attributes_randomly = True

    # Generate the dataset
    dataset = generate_dataset(
        num_samples,
        selected_entity_categories,
        sample_enitites_randomly,
        sample_attributes_randomly,
        selected_template_categories,
        templates_file=templates_file,
    )

    # Save the dataset
    save_dataset(dataset)

    # Print a few examples
    print("\nGenerated Dataset Examples:")
    for i, example in enumerate(dataset[:3]):
        print(f"\nExample {i+1}:")
        print(f"Context: {example['context_type']}")
        print(f"Text: {example['text']}")
