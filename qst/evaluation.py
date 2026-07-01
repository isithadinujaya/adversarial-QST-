from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from qst.attacks.frequency import pgd_frequency_attack
from qst.attacks.state import ID_TO_ATTACK
from qst.config import ExperimentConfig
from qst.quantum.metrics import fidelity, frobenius_distance, trace_distance


def _binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.to(torch.int64)
    positives = int(labels.sum())
    negatives = int(labels.numel() - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    true_positives = torch.cumsum(sorted_labels, dim=0).float()
    false_positives = torch.cumsum(1 - sorted_labels, dim=0).float()
    tpr = torch.cat([torch.zeros(1), true_positives / positives])
    fpr = torch.cat([torch.zeros(1), false_positives / negatives])
    return float(torch.trapz(tpr, fpr))


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
) -> dict[str, object]:
    model.eval()
    aggregate = defaultdict(float)
    per_attack: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    per_attack_count = defaultdict(int)
    pgd_aggregate = defaultdict(float)
    pgd_count = 0
    all_logits = []
    all_labels = []
    total_count = 0

    for batch in tqdm(loader, desc="evaluation", leave=False):
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        target = batch["target_density"]

        with torch.no_grad():
            clean_prediction, _ = model(batch["clean_frequencies"])
            input_prediction, logits = model(batch["input_frequencies"])

            clean_frobenius = frobenius_distance(clean_prediction, target)
            clean_trace = trace_distance(clean_prediction, target)
            clean_fidelity = fidelity(clean_prediction, target)
            input_frobenius = frobenius_distance(input_prediction, target)
            input_trace = trace_distance(input_prediction, target)
            input_fidelity = fidelity(input_prediction, target)

        batch_size = target.shape[0]
        total_count += batch_size
        aggregate["clean_frobenius"] += float(clean_frobenius.sum().cpu())
        aggregate["clean_trace_distance"] += float(clean_trace.sum().cpu())
        aggregate["clean_fidelity"] += float(clean_fidelity.sum().cpu())
        aggregate["input_frobenius"] += float(input_frobenius.sum().cpu())
        aggregate["input_trace_distance"] += float(input_trace.sum().cpu())
        aggregate["input_fidelity"] += float(input_fidelity.sum().cpu())

        for attack_id in torch.unique(batch["attack_id"]):
            attack_value = int(attack_id.item())
            mask = batch["attack_id"] == attack_id
            count = int(mask.sum())
            per_attack_count[attack_value] += count
            per_attack[attack_value]["frobenius"] += float(input_frobenius[mask].sum().cpu())
            per_attack[attack_value]["trace_distance"] += float(input_trace[mask].sum().cpu())
            per_attack[attack_value]["fidelity"] += float(input_fidelity[mask].sum().cpu())

        all_logits.append(logits.cpu())
        all_labels.append(batch["attack_label"].cpu())

        pgd_config = config.attacks.pgd
        if pgd_config.enabled:
            pgd_input = pgd_frequency_attack(
                model,
                batch["clean_frequencies"],
                target,
                epsilon=pgd_config.epsilon,
                step_size=pgd_config.step_size,
                steps=pgd_config.steps,
                number_settings=config.quantum.number_settings,
                outcomes_per_setting=config.quantum.outcomes_per_setting,
                random_start=pgd_config.random_start,
                detection_evasion_weight=pgd_config.detection_evasion_weight,
            )
            with torch.no_grad():
                pgd_prediction, pgd_logits = model(pgd_input)
                pgd_frobenius = frobenius_distance(pgd_prediction, target)
                pgd_trace = trace_distance(pgd_prediction, target)
                pgd_fidelity = fidelity(pgd_prediction, target)
                pgd_detected = (torch.sigmoid(pgd_logits) >= 0.5).float()
            pgd_aggregate["frobenius"] += float(pgd_frobenius.sum().cpu())
            pgd_aggregate["trace_distance"] += float(pgd_trace.sum().cpu())
            pgd_aggregate["fidelity"] += float(pgd_fidelity.sum().cpu())
            pgd_aggregate["detection_recall"] += float(pgd_detected.sum().cpu())
            pgd_count += batch_size

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    predictions = (torch.sigmoid(logits) >= 0.5).float()
    true_positive = float(((predictions == 1) & (labels == 1)).sum())
    false_positive = float(((predictions == 1) & (labels == 0)).sum())
    false_negative = float(((predictions == 0) & (labels == 1)).sum())
    accuracy = float((predictions == labels).float().mean())
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    report: dict[str, object] = {
        name: value / max(total_count, 1) for name, value in aggregate.items()
    }
    report["detection"] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": _binary_auc(logits, labels),
    }
    report["per_attack"] = {
        ID_TO_ATTACK[attack_id]: {
            metric: value / per_attack_count[attack_id]
            for metric, value in metrics.items()
        }
        for attack_id, metrics in sorted(per_attack.items())
    }
    if pgd_count > 0:
        report["pgd"] = {
            name: value / pgd_count for name, value in pgd_aggregate.items()
        }
        report["pgd"]["epsilon"] = config.attacks.pgd.epsilon
        report["pgd"]["steps"] = config.attacks.pgd.steps
    report["sample_count"] = total_count
    return report
