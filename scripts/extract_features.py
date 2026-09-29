"""Tile whole-slide images and extract patch features with a frozen encoder.

For each slide: find tissue, cut 256 x 256 patches at 20x (0.5 microns/pixel),
run every patch through a frozen pathology foundation model, and save one
HDF5 file per patient with the features and patch coordinates (for heatmaps).

Resumable: slides that already have a features file are skipped.

    python scripts/extract_features.py \
        --slides /path/to/slides \
        --cases data/matched_cases.csv \
        --out features/phikon \
        --encoder phikon

Encoders:
    uni      UNI (MahmoodLab), 1024-d. Needs Hugging Face access + login.
    phikon   Phikon (Owkin), 768-d. Open, no approval needed.
    resnet50 ImageNet ResNet50, 2048-d. Last-resort fallback.
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

TARGET_MPP = 0.5        # microns per pixel = 20x
PATCH = 256             # patch size in pixels at TARGET_MPP
MIN_TISSUE = 0.5        # keep patches that are at least half tissue
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ---------------------------------------------------------------- slides
def slide_mpp(slide):
    """Microns per pixel at level 0, from metadata (TCGA: 0.25 at 40x)."""
    mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp:
        return float(mpp)
    power = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if power:
        return 10.0 / float(power)  # 40x -> 0.25, 20x -> 0.5
    return 0.25  # most TCGA diagnostic slides are 40x


def tissue_coords(slide, patch_level0):
    """Level-0 top-left corners of patches that contain enough tissue."""
    width, height = slide.dimensions
    scale = 64  # thumbnail is 1/64 of level 0
    thumb = np.asarray(slide.get_thumbnail((width // scale, height // scale)).convert("RGB"))
    sx, sy = width / thumb.shape[1], height / thumb.shape[0]

    # Tissue is coloured (high saturation); glass is white/grey
    saturation = rgb2hsv(thumb)[..., 1]
    threshold = max(threshold_otsu(saturation), 0.05)
    mask = (saturation > threshold) & (thumb.mean(axis=2) > 30)  # drop black pen/edges

    coords = []
    for y in range(0, height - patch_level0 + 1, patch_level0):
        for x in range(0, width - patch_level0 + 1, patch_level0):
            x0, x1 = int(x / sx), max(int((x + patch_level0) / sx), int(x / sx) + 1)
            y0, y1 = int(y / sy), max(int((y + patch_level0) / sy), int(y / sy) + 1)
            if mask[y0:y1, x0:x1].mean() >= MIN_TISSUE:
                coords.append((x, y))
    return np.array(coords, dtype=np.int64).reshape(-1, 2)


class PatchDataset(Dataset):
    """Reads patches straight from the slide file (one handle per worker)."""

    def __init__(self, path, coords, patch_level0, transform):
        self.path, self.coords = path, coords
        self.patch_level0, self.transform = patch_level0, transform
        self.slide = None

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.path)
        slide = self.slide
        # Read from the pyramid level closest to 20x, then resize to PATCH
        downsample = self.patch_level0 / PATCH
        level = slide.get_best_level_for_downsample(downsample + 1e-6)
        level_size = int(round(self.patch_level0 / slide.level_downsamples[level]))
        x, y = self.coords[i]
        image = slide.read_region((int(x), int(y)), level, (level_size, level_size)).convert("RGB")
        if image.size != (PATCH, PATCH):
            image = image.resize((PATCH, PATCH), Image.BILINEAR)
        return self.transform(image)


# ---------------------------------------------------------------- encoders
def load_encoder(name, device):
    """Return (model, output_dim). Models are frozen and in eval mode."""
    if name == "uni":
        import timm
        model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True,
                                  init_values=1e-5, dynamic_img_size=True)
        forward, dim = model, 1024
    elif name == "phikon":
        from transformers import ViTModel
        vit = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        model = vit
        forward = lambda x: vit(pixel_values=x).last_hidden_state[:, 0, :]  # CLS token
        dim = 768
    elif name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        forward, dim = model, 2048
    else:
        raise ValueError(f"Unknown encoder: {name}")

    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return forward, dim


def find_slide(slide_dir, file_name):
    """gdc-client saves each slide as <slide_dir>/<file_id>/<file_name>."""
    matches = list(Path(slide_dir).rglob(file_name))
    return matches[0] if matches else None


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slides", required=True, help="folder gdc-client downloaded into")
    parser.add_argument("--cases", default="data/matched_cases.csv")
    parser.add_argument("--out", default="features/phikon")
    parser.add_argument("--encoder", default="phikon", choices=["uni", "phikon", "resnet50"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-patches", type=int, default=6000,
                        help="random cap per slide to bound run time (0 = no cap)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Encoder: {args.encoder}")

    forward, dim = load_encoder(args.encoder, device)
    transform = transforms.Compose([
        transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    cases = pd.read_csv(args.cases)
    done = skipped = 0
    for row in cases.itertuples():
        target = out / f"{row.patient}.h5"
        if target.exists():
            done += 1
            continue
        path = find_slide(args.slides, row.file_name)
        if path is None:
            skipped += 1
            continue  # not downloaded yet; re-run later to pick it up

        start = time.time()
        try:
            slide = openslide.OpenSlide(str(path))
            patch_level0 = int(round(PATCH * TARGET_MPP / slide_mpp(slide)))
            coords = tissue_coords(slide, patch_level0)
            slide.close()
        except Exception as error:
            print(f"  ! {row.patient}: could not read slide ({error})")
            continue

        if len(coords) == 0:
            print(f"  ! {row.patient}: no tissue found")
            continue
        if args.max_patches and len(coords) > args.max_patches:
            keep = np.random.default_rng(0).choice(len(coords), args.max_patches, replace=False)
            coords = coords[np.sort(keep)]

        loader = DataLoader(PatchDataset(str(path), coords, patch_level0, transform),
                            batch_size=args.batch_size, num_workers=args.workers,
                            pin_memory=device.type == "cuda")
        features = np.zeros((len(coords), dim), dtype=np.float32)
        position = 0
        with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
            for batch in loader:
                output = forward(batch.to(device, non_blocking=True)).float().cpu().numpy()
                features[position:position + len(output)] = output
                position += len(output)

        temp = target.with_suffix(".h5.tmp")
        with h5py.File(temp, "w") as h5:
            h5.create_dataset("features", data=features)
            h5.create_dataset("coords", data=coords)
            h5.attrs.update({"patient": row.patient, "slide": row.file_name,
                             "patch_size_level0": patch_level0, "encoder": args.encoder})
        temp.rename(target)  # only complete files count as done

        done += 1
        print(f"  {row.patient}: {len(coords)} patches in {time.time() - start:.0f}s "
              f"[{done}/{len(cases)}]", flush=True)

    print(f"\nFinished: {done} slides with features, {skipped} not downloaded yet.")


if __name__ == "__main__":
    main()
