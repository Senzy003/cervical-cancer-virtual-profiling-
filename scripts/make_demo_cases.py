"""Build demo cases for the UI from held-out (cross-validation) results.

For each chosen patient it writes demo_cases/<patient>/:
    thumbnail.png                slide overview
    heatmap_<protein>.png        attention overlay on the thumbnail
    top_<protein>_<k>.png        the patches the model weighted most
    result.json                  predictions, true values, image paths

Every prediction shown is from the fold where that patient was held out,
so the demo never shows a patient the model was trained on.

    python scripts/make_demo_cases.py --results results/phikon \
        --features features/phikon --slides slides/ --n 3
    # or choose patients yourself:
    python scripts/make_demo_cases.py ... --patients TCGA-C5-A1M5 TCGA-C5-A0TN
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
from matplotlib import colormaps
from PIL import Image

THUMB_WIDTH = 1600
TOP_K = 4
DISCLAIMER = ("Research prototype for decision support only. "
              "Not a diagnosis; confirm with laboratory testing.")


def safe(name):
    """'Cyclin B1' -> 'Cyclin_B1' for file names."""
    return re.sub(r"[^A-Za-z0-9-]+", "_", name)


def find_slide(slide_dir, file_name):
    matches = list(Path(slide_dir).rglob(file_name))
    return matches[0] if matches else None


def percentile_scores(attention):
    """Rank-normalise attention to 0..1 so the colours use the full range."""
    ranks = attention.argsort().argsort()
    return ranks / max(len(attention) - 1, 1)


def heatmap_overlay(thumb, coords, scores, patch_level0, scale, alpha=0.5):
    """Paint each patch's score onto the thumbnail and blend."""
    h, w = thumb.shape[:2]
    total = np.zeros((h, w), np.float32)
    count = np.zeros((h, w), np.float32)
    size = max(int(round(patch_level0 * scale)), 1)
    for (x, y), score in zip(coords, scores):
        x0, y0 = int(x * scale), int(y * scale)
        total[y0:y0 + size, x0:x0 + size] += score
        count[y0:y0 + size, x0:x0 + size] += 1
    covered = count > 0
    values = np.zeros_like(total)
    values[covered] = total[covered] / count[covered]
    colours = (colormaps["inferno"](values)[..., :3] * 255).astype(np.float32)
    out = thumb.astype(np.float32)
    out[covered] = (1 - alpha) * out[covered] + alpha * colours[covered]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def choose_patients(predictions, targets, n):
    """Pick clear examples: mostly-correct cases, plus one with a clear miss
    so the demo is honest about errors."""
    correct = pd.DataFrame(index=predictions.index)
    for t in targets:
        truth, prob = predictions[f"{t}_true"], predictions[f"{t}_abmil"]
        correct[t] = np.where(truth.isna(), np.nan, ((prob >= 0.5) == (truth == 1)).astype(float))
    confidence = sum((predictions[f"{t}_abmil"] - 0.5).abs() for t in targets)
    score = correct.mean(axis=1) + 0.1 * confidence
    ranked = score.sort_values(ascending=False).index.tolist()
    chosen = ranked[:max(n - 1, 1)]
    # Honest-but-fair "miss" case: the best-ranked patient with exactly one
    # wrong call among proteins that have a lab value (falls back to any miss).
    wrong = (correct == 0).sum(axis=1)
    one_miss = [p for p in ranked if wrong[p] == 1 and p not in chosen]
    any_miss = [p for p in ranked if wrong[p] >= 1 and p not in chosen]
    miss = one_miss or any_miss
    return chosen + miss[:1] if n > 1 else chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results/phikon")
    parser.add_argument("--features", default="features/phikon")
    parser.add_argument("--slides", default="slides")
    parser.add_argument("--cases", default="data/matched_cases.csv")
    parser.add_argument("--out", default="demo_cases")
    parser.add_argument("--n", type=int, default=3, help="number of demo cases to pick")
    parser.add_argument("--patients", nargs="*", help="specific patients instead of auto-pick")
    args = parser.parse_args()

    results = Path(args.results)
    config = json.loads((results / "config.json").read_text())
    targets = config["targets"]
    predictions = pd.read_csv(results / "oof_predictions.csv", index_col="patient")
    slide_files = pd.read_csv(args.cases).set_index("patient")["file_name"]
    metrics = pd.read_csv(results / "metrics.csv")
    abmil = metrics[metrics.model == "abmil"].set_index("protein")
    auc, ci_low = abmil["auc"].to_dict(), abmil["ci_low"].to_dict()

    patients = args.patients or choose_patients(predictions, targets, args.n)
    print("Demo cases:", ", ".join(patients))

    for patient in patients:
        slide_path = find_slide(args.slides, slide_files[patient])
        if slide_path is None:
            print(f"  ! {patient}: slide not found, skipping")
            continue
        folder = Path(args.out) / patient
        folder.mkdir(parents=True, exist_ok=True)

        attn = np.load(results / "attention" / f"{patient}.npz")
        attention, coords = attn["attention"], attn["coords"]
        with h5py.File(Path(args.features) / f"{patient}.h5", "r") as h5:
            patch_level0 = int(h5.attrs["patch_size_level0"])

        slide = openslide.OpenSlide(str(slide_path))
        width, height = slide.dimensions
        thumb_img = slide.get_thumbnail((THUMB_WIDTH, int(THUMB_WIDTH * height / width))).convert("RGB")
        thumb_img.save(folder / "thumbnail.png")
        thumb = np.asarray(thumb_img)
        scale = thumb.shape[1] / width

        cards = []
        for t, name in enumerate(targets):
            scores = percentile_scores(attention[:, t])
            heatmap_overlay(thumb, coords, scores, patch_level0, scale).save(
                folder / f"heatmap_{safe(name)}.png")

            tops = []
            for k, i in enumerate(np.argsort(attention[:, t])[::-1][:TOP_K], start=1):
                x, y = coords[i]
                patch = slide.read_region((int(x), int(y)), 0, (patch_level0, patch_level0))
                patch.convert("RGB").resize((256, 256)).save(folder / f"top_{safe(name)}_{k}.png")
                tops.append(f"/static/{patient}/top_{safe(name)}_{k}.png")

            prob = float(predictions.loc[patient, f"{name}_abmil"])
            truth = predictions.loc[patient, f"{name}_true"]
            cards.append({
                "protein": name,
                "call": "High" if prob >= 0.5 else "Low",
                "probability_high": round(prob, 3),
                "true_value": None if pd.isna(truth) else ("High" if truth == 1 else "Low"),
                "model_auc": None if pd.isna(auc.get(name)) else round(float(auc[name]), 2),
                "auc_ci_low": None if pd.isna(ci_low.get(name)) else round(float(ci_low[name]), 2),
                # Stronger evidence only when the whole 95% CI is above chance
                "evidence": ("stronger" if not pd.isna(ci_low.get(name)) and ci_low[name] > 0.5
                             else "exploratory"),
                "heatmap": f"/static/{patient}/heatmap_{safe(name)}.png",
                "top_patches": tops,
            })
        slide.close()

        # Stronger-evidence proteins first, then the rest
        cards.sort(key=lambda c: (c["evidence"] != "stronger", -(c["model_auc"] or 0)))
        with_lab = [c for c in cards if c["true_value"]]
        correct = sum(c["true_value"] == c["call"] for c in with_lab)
        result = {
            "case_id": patient,
            "summary": (f"Held-out case: {correct}/{len(with_lab)} lab-confirmed proteins match"
                        if with_lab else "Held-out case: no clear-cut lab values"),
            "thumbnail": f"/static/{patient}/thumbnail.png",
            "predictions": cards,
            "disclaimer": DISCLAIMER,
        }
        (folder / "result.json").write_text(json.dumps(result, indent=2))
        print(f"  {patient}: {result['summary']}")

    card = {
        "encoder": config.get("encoder", Path(args.features).name),
        "split": config.get("split", "median"),
        "n_patients": config.get("n_patients"),
        "folds": config.get("folds", 5),
        "metrics": metrics.replace({np.nan: None}).to_dict(orient="records"),
    }
    (Path(args.out) / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"\nDemo cases written to {args.out}/. Copy this folder to the demo laptop.")


if __name__ == "__main__":
    main()
