import torch
from nnsight import LanguageModel
from utils.activation import load_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
import json
from typing import Dict, Any, List, Optional
import numpy as np


torch.set_grad_enabled(False)


def is_equal(a: str, b: str) -> bool:
    f = lambda x: x.lower().strip("_ ")
    return f(a) == f(b)


def sample_to_encoding(sample: Dict[str, Any], model: LanguageModel) -> BatchEncoding:
    token_ids = sample["token_ids"][: -sample["num_right_pad"]]
    attention_mask = sample["attention_masks"][: -sample["num_right_pad"]]
    encoding = BatchEncoding(
        {
            "input_ids": torch.tensor([token_ids], device=model.device),
            "attention_mask": torch.tensor([attention_mask], device=model.device),
        }
    )
    return encoding


def generate_next_token(model: LanguageModel, sample: Dict[str, Any]) -> str:
    encoding = sample_to_encoding(sample, model)
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


def generate_with_intervention(
    model,
    base,
    layers,
    intervention_mode,
    source_act_LD=None,
    pin_direction_DC=None,
    pin_factor_LC=None,
):
    """add a diff vector at selected layers"""

    if (intervention_mode == "pin") and ((pin_direction_DC == None) or (pin_factor_LC == None)):
        raise ValueError("pin_direction and pin_factor must be set for pin_direction mode")
    if ((intervention_mode == "add") or (intervention_mode == "full_replace")) and (
        source_act_LD == None
    ):
        raise ValueError("source_act_LD must be set for add or full_replace mode")

    base_intervention_pos = int(base["key_to_position"]["a_question"])
    encoding = sample_to_encoding(base, model)

    with model.generate(encoding, max_new_tokens=1):
        for layer_idx, layer in enumerate(layers):
            resid_BPD = model.model.layers[layer].output[0]
            if intervention_mode == "add":
                resid_BPD[:, base_intervention_pos, :] += source_act_LD[layer_idx]
            elif intervention_mode == "pin":
                current_proj_BC = resid_BPD[:, base_intervention_pos, :] @ pin_direction_DC
                current_proj_BDC = current_proj_BC[:, None, :] * pin_direction_DC[None, :, :]
                target_proj_BDC = pin_factor_LC[None, None, layer, :] * pin_direction_DC[None, :, :]
                resid_BPD[:, base_intervention_pos, :] -= current_proj_BDC.sum(dim=-1)
                resid_BPD[:, base_intervention_pos, :] += target_proj_BDC.sum(dim=-1)
            elif intervention_mode == "full_replace":
                resid_BPD[:, base_intervention_pos, :] = source_act_LD[layer_idx]
            else:
                raise ValueError("intervention_mode must be one of 'add', 'pin', 'full_replace'")
            model.model.layers[layer].output = (resid_BPD,)
        generation = model.generator.output.save()
    generation = model.tokenizer.decode(generation[0, -1])

    return generation


def evaluate_intervention(generation, base, source):
    if is_equal(generation, base["special_str"]["base_object_from_source_box_pointer"]):
        return np.array([1, 0, 0, 0])
    elif is_equal(generation, base["special_str"]["correct_base_object"]):
        return np.array([0, 1, 0, 0])
    elif any([is_equal(generation, word) for word in base["key_to_word"].values()]):
        return np.array([0, 0, 1, 0])
    else:  # total_mismatch
        return np.array([0, 0, 0, 1])


evaluation_labels = [
    "targeted_object",
    "correct_object",
    "other_objects",
    "total_mismatch",
]


if __name__ == "__main__":
    from utils.dataset import gen_and_eval

    model_id = "google/gemma-2-2b"
    model = load_model(model_id)
    processed_dataset_path = "data/base_source_dataset_box_spo25.json"

    with open(processed_dataset_path, "r") as f:
        dataset = json.load(f)

    base_sample = dataset["base_dataset"][0]
    gen_eval_out = gen_and_eval(model, base_sample)

    inv_out = generate_with_intervention(
        model,
        base=base_sample,
        source_act_LD=None,
        layers=[],
        intervention_mode="pin",
        pin_direction_DC=torch.randn(model._model.config.hidden_size),
        pin_factor_LC=1,
    )

    inv_gen_eval = is_equal(base_sample["question"]["answer_token_str"], inv_out)

    print(f"Original: {gen_eval_out}, Intervention: {inv_gen_eval}")
