"""Detect galaxies in an LSST deep_coadd with a trained U-Net. Needs the LSST stack and Butler access (e.g. the Rubin
Science Platform).

    # a whole patch
    python scripts/detect_galaxies.py --model mep_unet --models-root ~/unet_models --tract 2877 --patch 34 --out detections.parquet
    # a 10' square around a position
    python scripts/detect_galaxies.py --model mep_unet --models-root ~/unet_models --ra 59.5 --dec -0.75 --size 3072 --out cirrus.parquet

The catalogue has pixel (x, y), tract pixel (x_tract, y_tract) and sky (ra, dec) positions, p_real and raw_score.
"""

import argparse
from pathlib import Path

from lsst_unet_detection import BANDS, CONFIG, cutout_bbox, detect_galaxies, load_deep_coadds, load_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model folder, or its name inside --models-root")
    parser.add_argument("--models-root", type=Path, help="folder holding one sub-folder per trained model")
    parser.add_argument("--out", type=Path, required=True, help="output catalogue (.parquet or .csv)")
    where = parser.add_argument_group("where to look: --tract and --patch, or --ra, --dec and --size")
    where.add_argument("--tract", type=int)
    where.add_argument("--patch", type=int)
    where.add_argument("--ra", type=float, help="centre RA (deg)")
    where.add_argument("--dec", type=float, help="centre Dec (deg)")
    where.add_argument("--size", type=int, default=3072, help="cutout side in pixels (default: %(default)s)")
    parser.add_argument("--repo", default=CONFIG["butler_repo"], help="Butler repository (default: %(default)s)")
    parser.add_argument("--collections", default=CONFIG["collections"], help="default: %(default)s")
    parser.add_argument("--skymap", default=CONFIG["skymap"], help="default: %(default)s")
    parser.add_argument("--threshold", type=float, help="override the model's calibrated p_real threshold")
    args = parser.parse_args()
    by_patch = args.tract is not None and args.patch is not None
    by_position = args.ra is not None and args.dec is not None
    if by_patch == by_position:
        parser.error("give either --tract and --patch, or --ra and --dec")

    from lsst.daf.butler import Butler  # pyright: ignore[reportMissingImports]  (LSST stack)

    detector = load_detector(args.model, args.models_root, args.threshold)
    butler = Butler(args.repo, collections=args.collections)
    if by_patch:
        tract, patch, bbox = args.tract, args.patch, None
    else:
        tract, patch, bbox = cutout_bbox(butler, args.ra, args.dec, args.size, args.skymap)
    print(f"Loading {''.join(BANDS)} deep_coadd for tract {tract}, patch {patch}" + (f", {bbox}" if bbox else ""))
    coadds = load_deep_coadds(butler, tract, patch, bbox=bbox, skymap=args.skymap)
    detections = detect_galaxies(coadds, detector).assign(tract=tract, patch=patch, model=detector.name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".parquet":
        detections.to_parquet(args.out)
    else:
        detections.to_csv(args.out, index=False)
    print(f"{len(detections):,} detections with p_real >= {detector.threshold:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
