"""Download a published trained model into a models folder, ready to load by name.

    python scripts/download_model.py --list
    python scripts/download_model.py --model mep_unet_v1 --models-root ~/unet_models
    python scripts/detect_galaxies.py --model mep_unet_v1 --models-root ~/unet_models --coadd cirrus.npz --out cirrus.parquet
"""

import argparse
from pathlib import Path

from lsst_unet_detection import PUBLISHED_MODELS, download_model


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="name of a published model")
    parser.add_argument("--models-root", type=Path, help="folder holding one sub-folder per trained model")
    parser.add_argument("--list", action="store_true", help="list the published models")
    args = parser.parse_args()
    if args.list:
        for name, entry in PUBLISHED_MODELS.items():
            print(f"{name}: {entry['about']}")
        return
    if args.model is None or args.models_root is None:
        parser.error("give --model and --models-root (or --list)")
    download_model(args.model, args.models_root)


if __name__ == "__main__":
    main()
