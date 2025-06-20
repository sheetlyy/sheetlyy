from pathlib import Path
import os
import requests
import zipfile
import cv2

GITHUB_ASSETS_URL = "https://api.github.com/repos/sheetlyy/sheetlyy/releases/latest"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")


def get_models():
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
    }

    res = requests.get(GITHUB_ASSETS_URL, headers=headers)
    res.raise_for_status()
    assets = res.json().get("assets")
    return {asset["name"]: asset["url"] for asset in assets}


def download_model(url: str, filename: Path) -> None:
    headers = {
        "Accept": "application/octet-stream",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
    }

    res = requests.get(url, headers=headers, stream=True, timeout=5)
    res.raise_for_status()

    # Download model
    filename.parent.mkdir(exist_ok=True, parents=True)
    with open(filename, "wb") as f:
        for chunk in res.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip_model(filename: Path, output_folder: Path) -> None:
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_contents = zip_ref.namelist()
        for member in zip_contents:
            # Ensure file path is safe
            if Path(member).is_absolute() or ".." in Path(member).parts:
                print(f"Skipping potentially unsafe file {member}")
                continue

            # Handle directories
            if member.endswith("/"):
                output_folder.joinpath(member).mkdir(exist_ok=True, parents=True)
                continue

            # Extract file
            source = zip_ref.open(member)
            target = open(output_folder.joinpath(member), "wb")

            with source, target:
                while True:
                    chunk = source.read(1024)
                    if not chunk:
                        break
                    target.write(chunk)


def download_weights() -> None:
    models_dir = Path.cwd().joinpath("models")
    if models_dir.exists():
        return

    models = get_models()

    print("Downloading models")
    for zip_name, model_url in models.items():
        try:
            zip_path = models_dir.joinpath(zip_name)

            download_model(model_url, zip_path)
            unzip_model(zip_path, models_dir)
        finally:
            if zip_path.exists():
                zip_path.unlink()
    print("Downloaded all models")


download_weights()
