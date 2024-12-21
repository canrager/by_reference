import torch
from nnsight import LanguageModel
from activation_utils import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
import json
from typing import Dict, Any, List, Optional

torch.set_grad_enabled(False)


def is_equal(a: str, b: str) -> bool:
    f = lambda x: x.lower().strip("_ ")
    return f(a) == f(b)


def generate_next_token(model: LanguageModel, sample: Dict[str, Any]) -> str:
    token_ids = sample["token_ids"][: -sample["num_right_pad"]]
    attention_mask = sample["attention_masks"][: -sample["num_right_pad"]]
    encoding = BatchEncoding(
        {
            "input_ids": torch.tensor([token_ids], device=model.device),
            "attention_mask": torch.tensor([attention_mask], device=model.device),
        }
    )
    # generation = model.generate(token_ids, max_new_tokens=1)
    with model.generate(encoding, max_new_tokens=1):
        generation = model.generator.output.save()
    generation = model.tokenizer.decode(generation[0, -1])
    return generation


def gen_and_eval(model: LanguageModel, sample: Dict[str, Any]) -> bool:
    correct_answer = sample["question"]["answer_token_str"]
    pred_str = generate_next_token(model, sample)
    is_correct = is_equal(correct_answer, pred_str)
    return is_correct


def filter_correct(
    model: LanguageModel, dataset: List[Dict[str, Any]], max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    filtered_ds = [sample for sample in dataset if gen_and_eval(model, sample)]
    accuracy = len(filtered_ds) / len(dataset)
    if max_samples:
        filtered_ds = filtered_ds[:max_samples]
        if len(filtered_ds) < max_samples:
            print(f"Warning: Only {len(filtered_ds)} samples found.")
    return filtered_ds, accuracy


# def check_accuracy(model: LanguageModel, dataset: List[Dict[str, Any]]) -> float:
#     correct = 0
#     for sample in dataset:
#         correct_answer = sample["question"]["answer_token_str"]
#         pred_str = generate_next_token(model, sample)
#         is_correct = is_equal(correct_answer, pred_str)
#         if not is_correct:
#             print(f"Correct: {correct_answer}, Predicted: {pred_str}")
#         correct += is_correct
#     print(f"Correct: {correct}, Total: {len(dataset)}")
#     return correct / len(dataset)


def main():
    model_id = "google/gemma-2-2b"
    # model = load_model(model_id)
    model = load_model(model_id)
    processed_dataset_path = "data/processed_dataset.json"

    with open(processed_dataset_path, "r") as f:
        dataset = json.load(f)

    num_samples = 100
    acc = check_accuracy(model, dataset[:num_samples])
    print(f"Accuracy: {acc:.2f}")

    # Baseline
    num_entities = len(dataset[0]["entity_keys"])
    baseline_acc = 1 / num_entities
    print(f"Baseline accuracy: {baseline_acc:.2f}")


if __name__ == "__main__":
    main()
