"""Report binary spleen Dice with equal weight for each labeled patient."""
import json
from statistics import mean


def save_dice_scores(output_dir, scores):
    if not scores:
        return
    report = {"per_patient_dice": scores}
    if len(scores) > 1:
        report["mean_dice"] = mean(scores.values())
        print(f"Mean Dice across {len(scores)} patients: {report['mean_dice']:.4f}")
    (output_dir / "dice_scores.json").write_text(json.dumps(report, indent=2) + "\n")
