"""Trained models published as GitHub release assets, and their download.

Each published model is a zip of one model folder (weights, normalisation, model config, calib peaks, threshold),
attached to a release of this repository. download_model fetches it into a models folder, checks its SHA-256 and
unpacks it, so it can then be loaded by name like any other model:

    download_model("mep_unet_v1", "~/unet_models")
    detector = load_detector("mep_unet_v1", models_root="~/unet_models")
"""

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .config import ARTEFACTS

RELEASES = "https://github.com/tmsedgwick/lsst_unet_detection/releases/download"

PUBLISHED_MODELS = {
    "mep_unet_v1": dict(
        url=f"{RELEASES}/mep_unet_v1/mep_unet_v1.zip",
        sha256="21aae27833b3a08a4afa9aadc852b06a9d9c41b24eea0a19b00d98d40202a899",
        about="six-band U-Net (ugrizy), threshold for 99% purity on the July mocks; no band adapter",
    ),
}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_model(name, models_root, published=None):
    """Download the published model name into <models_root>/<name> and return that folder. A model already there is
    left as it is."""
    published = PUBLISHED_MODELS if published is None else published
    if name not in published:
        raise KeyError(f"No published model {name!r}; published: {', '.join(sorted(published)) or 'none'}")
    target = Path(models_root).expanduser() / name
    if (target / ARTEFACTS["weights"]).exists():
        print(f"{name} is already in {target.parent}")
        return target
    entry = published[name]
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        archive = Path(tmp) / f"{name}.zip"
        print(f"Downloading {name} from {entry['url']}")
        with urllib.request.urlopen(entry["url"]) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out)
        if sha256_of(archive) != entry["sha256"]:
            raise ValueError(f"{name}: the download's SHA-256 does not match the published one; not installed")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(tmp)
        unpacked = Path(tmp) / name
        missing = [file for file in ARTEFACTS.values() if not (unpacked / file).exists()]
        if missing:
            raise ValueError(f"{name}: the download lacks {', '.join(missing)}; not installed")
        unpacked.rename(target)
    print(f"Installed {name} in {target}")
    return target
