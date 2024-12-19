from dataset_utils import generate_dataset, process_dataset, print_processed_example
from transformers import AutoTokenizer

# Dataset generation parameters
DATASET_PARAMS = {
    "num_samples": 100,
    "sample_entities_randomly": True,
    "sample_attributes_randomly": True,
    "selected_template_categories": ["box_templates"],
    "templates_file": "data/templates_box.json",
    "entities_file": "data/entities.json",
    "attributes_file": "data/attributes.json",
    "raw_dataset_path": "data/dataset.json"
}

# Experiment parameters
EXPERIMENT_PARAMS = {
    "model_id": "google/gemma-2-2b",
    "max_context_length": 100,
    "total_length": 1000, # Pre- and append padding tokens to the text filling up to total_length
    "random_offset": True,
    "processed_dataset_path": "data/processed_dataset.json"
}

def main():
    # Generate dataset
    print("Generating dataset...")
    dataset = generate_dataset(**DATASET_PARAMS)

    # Print examples
    print("\nGenerated Dataset Examples:")
    for i, example in enumerate(dataset[:3]):
        print(f"\nExample {i+1}:")
        print(f"Context: {example['context_type']}")
        print(f"Text: {example['text']}")
    
    # Process and tokenize dataset
    print("Processing and tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained(EXPERIMENT_PARAMS["model_id"])
    processed_dataset = process_dataset(
        dataset,
        tokenizer,
        max_context_length=EXPERIMENT_PARAMS["max_context_length"],
        total_length=EXPERIMENT_PARAMS["total_length"],
        random_offset=EXPERIMENT_PARAMS["random_offset"],
        save_path=EXPERIMENT_PARAMS["processed_dataset_path"]
    )
    print(f"Successfully processed {len(processed_dataset)} examples")

    print_processed_example(processed_dataset[0], tokenizer)

if __name__ == "__main__":
    main()