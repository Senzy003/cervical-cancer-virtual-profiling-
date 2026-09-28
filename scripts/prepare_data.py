"""Prepare TCGA-CESC data: find patients with BOTH a diagnostic H&E slide and
RPPA protein data, write a gdc-client manifest for their slides, download the
protein files and build one patient x protein table.

Run on the cluster (login node is fine; it only downloads small files):

    python scripts/prepare_data.py --out data

Outputs (in --out):
    manifests/slides_manifest.txt   gdc-client manifest, matched patients only
    matched_cases.csv               patient, slide file, RPPA file, slide size
    rppa_matrix.csv                 patients x proteins (RPPA values)
    rppa_proteins.txt               every protein name in the RPPA panel
    target_report.txt               which of our candidate targets exist
"""

import argparse
import io
import re
import sys
import tarfile
from pathlib import Path

import pandas as pd
import requests

GDC = "https://api.gdc.cancer.gov"
PROJECT = "TCGA-CESC"

# Candidate targets and patterns to find them in RPPA peptide_target names.
TARGETS = {
    "PD-L1": r"pd-?l1|cd274",
    "p16": r"p16|cdkn2a",
    "Ki-67": r"ki-?67|mki67",
    "p53": r"^p53$|tp53",
    "Rb": r"^rb$|^rb_",
    "phospho-Rb": r"rb_p",
    "Cyclin B1": r"cyclin.?b1|ccnb1",
    "PCNA": r"pcna",
    "E-cadherin": r"e-?cadherin|cdh1",
    "Fibronectin": r"fibronectin|fn1",
}

FILE_FIELDS = ["file_id", "file_name", "md5sum", "file_size", "state",
               "cases.submitter_id", "cases.samples.sample_type"]


def query_files(data_type, extra=None):
    """Return all TCGA-CESC files of one data type as a DataFrame."""
    filters = [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [PROJECT]}},
        {"op": "in", "content": {"field": "data_type", "value": [data_type]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}},
    ] + (extra or [])
    body = {"filters": {"op": "and", "content": filters},
            "fields": ",".join(FILE_FIELDS), "format": "JSON", "size": 5000}
    response = requests.post(f"{GDC}/files", json=body, timeout=120)
    response.raise_for_status()
    hits = response.json()["data"]["hits"]

    rows = []
    for hit in hits:
        case = hit["cases"][0]
        samples = case.get("samples") or [{}]
        rows.append({
            "file_id": hit["file_id"],
            "file_name": hit["file_name"],
            "md5sum": hit["md5sum"],
            "file_size": hit["file_size"],
            "state": hit.get("state", "released"),
            "patient": case["submitter_id"][:12],
            "sample_type": samples[0].get("sample_type", ""),
        })
    return pd.DataFrame(rows)


def download_rppa(file_ids, dest):
    """Download RPPA TSVs through the GDC data endpoint (tar.gz batches)."""
    dest.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(file_ids), 100):
        batch = file_ids[start:start + 100]
        response = requests.post(f"{GDC}/data", json={"ids": batch}, timeout=600)
        response.raise_for_status()
        if len(batch) == 1:
            (dest / f"{batch[0]}.tsv").write_bytes(response.content)
            continue
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
            for member in tar.getmembers():
                # Tar entries look like <file_id>/<file_name>; skip MANIFEST.txt
                if member.isfile() and member.name.count("/") == 1:
                    file_id = member.name.split("/")[0]
                    data = tar.extractfile(member).read()
                    (dest / f"{file_id}.tsv").write_bytes(data)
        print(f"  RPPA files downloaded: {min(start + 100, len(file_ids))}/{len(file_ids)}")


def build_matrix(rppa, folder):
    """Combine per-sample RPPA TSVs into one patients x proteins table."""
    columns = {}
    for row in rppa.itertuples():
        path = folder / f"{row.file_id}.tsv"
        if not path.exists():
            continue
        table = pd.read_csv(path, sep="\t")
        values = table.set_index("peptide_target")["protein_expression"]
        values = pd.to_numeric(values, errors="coerce")
        columns[row.patient] = values[~values.index.duplicated()]
    return pd.DataFrame(columns).T.sort_index()


def target_report(proteins):
    lines = []
    for target, pattern in TARGETS.items():
        found = [p for p in proteins if re.search(pattern, p, re.IGNORECASE)]
        status = "FOUND  " if found else "MISSING"
        lines.append(f"{status} {target:12} {', '.join(found)}")
    missing = [t for t, p in TARGETS.items()
               if not any(re.search(p, x, re.IGNORECASE) for x in proteins)]
    if missing:
        lines.append("\nMissing targets: use RNA-seq proxies (CD274, CDKN2A, "
                     "MKI67) or swap for a FOUND protein.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data", help="output folder")
    parser.add_argument("--skip-rppa-download", action="store_true",
                        help="reuse RPPA files already in <out>/rppa_raw")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "manifests").mkdir(parents=True, exist_ok=True)

    print("Querying GDC for diagnostic slides...")
    slides = query_files("Slide Image", [
        {"op": "in", "content": {"field": "experimental_strategy",
                                 "value": ["Diagnostic Slide"]}}])
    print(f"  {len(slides)} diagnostic slides, {slides.patient.nunique()} patients")

    print("Querying GDC for RPPA protein files...")
    rppa = query_files("Protein Expression Quantification")
    rppa = rppa[rppa.sample_type.str.contains("Tumor", case=False, na=True)]
    rppa = rppa.drop_duplicates("patient")  # one tumour profile per patient
    print(f"  {len(rppa)} RPPA files, {rppa.patient.nunique()} patients")

    matched_patients = sorted(set(slides.patient) & set(rppa.patient))
    if not matched_patients:
        sys.exit("No patients have both a slide and RPPA data; check the queries.")

    # One diagnostic slide per patient (the first; most patients have one)
    matched_slides = (slides[slides.patient.isin(matched_patients)]
                      .sort_values("file_name").drop_duplicates("patient"))
    size_gb = matched_slides.file_size.sum() / 1e9
    print(f"\nMatched patients: {len(matched_patients)}  "
          f"(slide download about {size_gb:.0f} GB)")

    manifest = matched_slides.rename(columns={"file_id": "id", "file_name": "filename",
                                              "md5sum": "md5", "file_size": "size"})
    manifest["size"] = manifest["size"].astype("int64")
    manifest[["id", "filename", "md5", "size", "state"]].to_csv(
        out / "manifests" / "slides_manifest.txt", sep="\t", index=False)

    matched = matched_slides[["patient", "file_name", "file_size"]].merge(
        rppa[["patient", "file_id"]].rename(columns={"file_id": "rppa_file_id"}),
        on="patient")
    matched.to_csv(out / "matched_cases.csv", index=False)

    raw = out / "rppa_raw"
    if not args.skip_rppa_download:
        print("\nDownloading RPPA files...")
        download_rppa(rppa.file_id.tolist(), raw)

    matrix = build_matrix(rppa, raw)
    matrix = matrix.loc[matrix.index.intersection(matched_patients)]
    matrix.to_csv(out / "rppa_matrix.csv")
    (out / "rppa_proteins.txt").write_text("\n".join(matrix.columns) + "\n")
    print(f"RPPA matrix: {matrix.shape[0]} patients x {matrix.shape[1]} proteins")

    report = target_report(list(matrix.columns))
    (out / "target_report.txt").write_text(report + "\n")
    print("\nCandidate targets:\n" + report)
    print(f"\nNext: gdc-client download -m {out}/manifests/slides_manifest.txt "
          f"-d <slide folder> -n 8")


if __name__ == "__main__":
    main()
