import requests
from pathlib import Path
import zipfile
import shutil
import os

# List of datasets: name, URLs, and target folder
datasets = [
    {
        "name": "Hemorrhagic_Dataset",
        "urls": [
            "https://acikveri.saglik.gov.tr/Document/Download/39",
            "https://acikveri.saglik.gov.tr/Document/Download/40",
            "https://acikveri.saglik.gov.tr/Document/Download/41",
            "https://acikveri.saglik.gov.tr/Document/Download/42",
            "https://acikveri.saglik.gov.tr/Document/Download/43",
            "https://acikveri.saglik.gov.tr/Document/Download/44",
        ],
        "target": Path("downloads/stroke2021/Training/Hemorrhagic"),
    },
    {
        "name": "Ischemic_Dataset",
        "urls": [
            "https://acikveri.saglik.gov.tr/Document/Download/33",
            "https://acikveri.saglik.gov.tr/Document/Download/34",
            "https://acikveri.saglik.gov.tr/Document/Download/35",
            "https://acikveri.saglik.gov.tr/Document/Download/36",
            "https://acikveri.saglik.gov.tr/Document/Download/37",
            "https://acikveri.saglik.gov.tr/Document/Download/38",
        ],
        "target": Path("downloads/stroke2021/Training/Ischemic"),
    },
    {
        "name": "NonStroke_Dataset_PNG",
        "urls": [
            "https://acikveri.saglik.gov.tr/Document/Download/26",
            "https://acikveri.saglik.gov.tr/Document/Download/27",
            "https://acikveri.saglik.gov.tr/Document/Download/28",
            "https://acikveri.saglik.gov.tr/Document/Download/29",
            "https://acikveri.saglik.gov.tr/Document/Download/30",
            "https://acikveri.saglik.gov.tr/Document/Download/31",
            "https://acikveri.saglik.gov.tr/Document/Download/32"
        ],
        "target": Path("downloads/stroke2021/Training/Non-Stroke/PNG"),
    },
    {
        "name": "NonStroke_Dataset_DICOM",
        "urls": [
            "https://acikveri.saglik.gov.tr/Document/Download/14",
            "https://acikveri.saglik.gov.tr/Document/Download/15",
            "https://acikveri.saglik.gov.tr/Document/Download/16",
            "https://acikveri.saglik.gov.tr/Document/Download/17",
            "https://acikveri.saglik.gov.tr/Document/Download/18",
            "https://acikveri.saglik.gov.tr/Document/Download/19",
            "https://acikveri.saglik.gov.tr/Document/Download/20",
            "https://acikveri.saglik.gov.tr/Document/Download/21",
            "https://acikveri.saglik.gov.tr/Document/Download/22",
            "https://acikveri.saglik.gov.tr/Document/Download/23",
        ],
        "target": Path("downloads/stroke2021/Training/Non-Stroke/DICOM"),
    },
]

# Base folder for downloads
# Base folder for downloads
base_download_dir = Path("downloads/compressed")
base_download_dir.mkdir(parents=True, exist_ok=True)

for dataset in datasets:
    print(f"\n--- {dataset['name']} ---")
    download_dir = base_download_dir / dataset["name"]
    download_dir.mkdir(parents=True, exist_ok=True)
    target_dir = dataset["target"]
    target_dir.mkdir(parents=True, exist_ok=True)

    merged_zip = download_dir / f"{dataset['name']}.zip"

    # If merged zip already exists, skip downloading parts
    if merged_zip.exists():
        print(f"Merged zip already exists, skipping download of parts: {merged_zip}")
    else:
        # Download part files
        for url in dataset["urls"]:
            filename = download_dir / Path(url).name
            if filename.exists():
                print(f"Already exists: {filename.name}")
                continue
            print(f"Downloading: {filename.name}")
            resp = requests.get(url, stream=True)
            with open(filename, "wb") as f:
                shutil.copyfileobj(resp.raw, f)

        # Merge parts into a single zip file
        with open(merged_zip, "wb") as outfile:
            for url in dataset["urls"]:
                part_file = download_dir / Path(url).name
                with open(part_file, "rb") as infile:
                    shutil.copyfileobj(infile, outfile)
        print("Parts merged successfully.")

        # Delete part files after merging
        for url in dataset["urls"]:
            part_file = download_dir / Path(url).name
            if part_file.exists():
                os.remove(part_file)
        print("Part files deleted.")

    # Extract zip without keeping the top-level folder
    with zipfile.ZipFile(merged_zip, "r") as zip_ref:
        top_level_folders = set()
        for member in zip_ref.namelist():
            parts = Path(member).parts
            if len(parts) > 1:
                top_level_folders.add(parts[0])  # remember top folder
            fixed_path = Path(*parts[1:]) if len(parts) > 1 else Path(member)
            if fixed_path:
                target_path = target_dir / fixed_path
                if member.endswith("/"):
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(member) as source, open(target_path, "wb") as target_file:
                        shutil.copyfileobj(source, target_file)

    # Clean up any leftover top-level folders (if created by extract)
    for folder in top_level_folders:
        candidate = target_dir / folder
        if candidate.exists() and candidate.is_dir() and not any(candidate.iterdir()):
            shutil.rmtree(candidate)
            print(f"Removed empty folder: {candidate}")

    print(f"Archive extracted to: {target_dir}")