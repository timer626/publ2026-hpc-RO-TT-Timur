
import re
import string
from collections import Counter

import torch
from torch import Tensor


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(
        char for char in text
        if char not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_em(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction)== normalize_answer(reference))


def answer_f1(prediction: str, reference: str) -> float:
    prediction = normalize_answer(prediction)
    reference = normalize_answer(reference)

    special_answers = {"yes", "no", "noanswer"}

    if prediction in special_answers or reference in special_answers:
        return float(prediction == reference)

    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def evaluate_answer(prediction: str,reference: str) -> dict[str, float]:
    return {"answer_em": answer_em(prediction, reference),"answer_f1": answer_f1(prediction, reference)}



def tt_reconstruction_error(original: Tensor,reconstructed: Tensor) -> float:
    
    original = original.float()
    reconstructed = reconstructed.float()

    numerator = torch.linalg.vector_norm((original - reconstructed).reshape(-1),ord=2,)
    denominator = torch.linalg.vector_norm(original.reshape(-1),ord=2,)

    return (numerator / denominator).item()

