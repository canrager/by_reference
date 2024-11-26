import json
import random
from typing import Dict, List, Any
import os

def load_json(filename: str) -> Dict:
    """Load a JSON file and return its contents."""
    with open(filename, 'r') as f:
        return json.load(f)

def get_random_entities(entities: Dict[str, List[str]], n: int = 3) -> List[str]:
    """Select n random entities, optionally from specific categories."""
    # Flatten all entity categories into one list
    all_entities = []
    for category in entities.values():
        all_entities.extend(category)
    return random.sample(all_entities, n)

def get_random_attributes(template: str, attributes: Dict[str, Dict[str, List[str]]]) -> Dict[str, str]:
    """Generate random attributes based on the template's requirements."""
    attr_dict = {}
    
    # Parse template to find required attributes
    # Look for patterns like {e1_color}, {e2_state}, etc.
    import re
    attr_patterns = re.findall(r'\{([^}]+)\}', template)
    
    # Remove entity references (e1, e2, e3)
    attr_patterns = [p for p in attr_patterns if p not in ['e1', 'e2', 'e3']]
    
    for attr_pattern in attr_patterns:
        # Split pattern into entity number and attribute type (e.g., 'e1_color' -> 'e1', 'color')
        entity, attr_type = attr_pattern.split('_')
        
        # Find the attribute category containing this attribute type
        for category, subcategories in attributes.items():
            for subcat, values in subcategories.items():
                if attr_type in subcat or any(attr_type in val for val in values):
                    attr_dict[attr_pattern] = random.choice(values)
                    break
            if attr_pattern in attr_dict:
                break
                
        # If not found in any category, use a default value
        if attr_pattern not in attr_dict:
            attr_dict[attr_pattern] = f"unknown_{attr_type}"
    
    return attr_dict

def generate_dataset(
    num_samples: int,
    entities_file: str = 'entities.json',
    attributes_file: str = 'attributes.json',
    templates_file: str = 'templates.json'
) -> List[Dict[str, Any]]:
    """Generate a dataset of formatted templates with random entities and attributes."""
    
    # Load the JSON files
    entities = load_json(entities_file)
    attributes = load_json(attributes_file)
    templates = load_json(templates_file)
    
    dataset = []
    
    for _ in range(num_samples):
        # Select random context and template
        context = random.choice(list(templates.keys()))
        template = random.choice(templates[context])
        
        # Get random entities
        entity_list = get_random_entities(entities)
        
        # Get random attributes based on template
        attr_dict = get_random_attributes(template, attributes)
        
        # Combine entities and attributes
        format_dict = {
            'e1': entity_list[0],
            'e2': entity_list[1],
            'e3': entity_list[2],
            **attr_dict
        }
        
        # Format the template
        formatted_text = template.format(**format_dict)
        
        # Store the example
        example = {
            'context_type': context,
            'template': template,
            'entities': entity_list,
            'attributes': attr_dict,
            'text': formatted_text
        }
        
        dataset.append(example)
    
    return dataset

def save_dataset(dataset: List[Dict[str, Any]], output_file: str = 'dataset.json'):
    """Save the generated dataset to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

def main():
    """Main function to generate and save the dataset."""
    # Generate 100 examples
    dataset = generate_dataset(100)
    
    # Save the dataset
    save_dataset(dataset)
    
    # Print a few examples
    print("\nGenerated Dataset Examples:")
    for i, example in enumerate(dataset[:3]):
        print(f"\nExample {i+1}:")
        print(f"Context: {example['context_type']}")
        print(f"Text: {example['text']}")

if __name__ == "__main__":
    main()
