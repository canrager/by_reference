import json
import random
from typing import Dict, List, Any

def load_json(filename: str) -> Dict:
    """Load a JSON file and return its contents."""
    with open(filename, 'r') as f:
        return json.load(f)

def get_random_entities(entities: List[str], n: int = 2) -> List[str]:
    """Select n random entities."""
    return random.sample(entities, n)

def get_random_attributes(n: int, attributes: Dict[str, str]) -> List[str]:
    """Get n random attributes (cities) and their corresponding countries."""
    countries = list(attributes.keys())
    selected_countries = random.sample(countries, n)
    return selected_countries, [attributes[country] for country in selected_countries]

def generate_dataset(
    num_samples: int,
    entities_file: str = 'entities.json',
    attributes_file: str = 'attributes.json',
    templates_file: str = 'templates.json'
) -> List[Dict[str, Any]]:
    """Generate a dataset with the specified format."""
    
    # Load the JSON files
    entities_data = load_json(entities_file)
    attributes = load_json(attributes_file)
    templates = load_json(templates_file)
    
    entities = entities_data["global_names"]
    dataset = []
    
    for _ in range(num_samples):
        # Get template
        template = templates["capital_city"][0]
        
        # Get random entities and attributes
        entity_list = get_random_entities(entities, 2)
        countries, cities = get_random_attributes(2, attributes)
        
        # Randomly select which entity to ask about
        qn_idx = random.randint(0, 1)
        qn_subject = entity_list[qn_idx]
        answer_city = cities[qn_idx]
        
        # Create the formatted text with qn_subject included
        format_dict = {
            'E_0': entity_list[0],
            'E_1': entity_list[1],
            'A_0': countries[0],
            'A_1': countries[1],
            'qn_subject': qn_subject  # Add qn_subject to format_dict
        }
        
        # Format the base text
        formatted_text = template.format(**format_dict)
        
        # Store the example
        example = {
            'context_type': 'capital_city',
            'template': template,
            'entities': entity_list,
            'attributes': {
                'countries': countries,
                'cities': cities
            },
            'text': formatted_text,
            'qn_subject': qn_subject,
            'answer': answer_city
        }
        
        dataset.append(example)
    
    return dataset

def save_dataset(dataset: List[Dict[str, Any]], output_file: str = 'dataset.json'):
    """Save the generated dataset to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)

def main():
    """Main function to generate and save the dataset."""
    # Generate examples
    dataset = generate_dataset(100)
    
    # Save the dataset
    save_dataset(dataset)
    
    # Print a few examples
    print("\nGenerated Dataset Examples:")
    for i, example in enumerate(dataset[:3]):
        print(f"\nExample {i+1}:")
        print(f"Text: {example['text']}")
        print(f"Question Subject: {example['qn_subject']}")
        print(f"Answer: {example['answer']}")

if __name__ == "__main__":
    main()