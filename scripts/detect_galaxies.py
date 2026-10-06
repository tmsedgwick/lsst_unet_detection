"""Detect galaxies in an LSST deep_coadd with a trained U-Net. Reading the coadd from the Butler needs the LSST stack
(e.g. the Rubin Science Platform); a coadd saved as .npz (--save-coadd) can be read anywhere.

    # a whole patch
    python scripts/detect_galaxies.py --model mep_unet --models-root ~/unet_models --tract 2877 --patch 34 --out detections.parquet
    # a 10' square around a position
    python scripts/detect_galaxies.py --model mep_unet --models-root ~/unet_models --ra 59.5 --dec -0.75 --size 3072 --out cirrus.parquet
    # a coadd saved as .npz, without the LSST stack
    python scripts/detect_galaxies.py --model mep_unet --models-root ~/unet_models --coadd cirrus.npz --out cirrus.parquet

The catalogue has pixel (x, y), tract pixel (x_tract, y_tract) and, from the Butler, sky (ra, dec) positions, raw_score and
p_detection_centroid (the probability that the detection is the centre of a real galaxy or star), and for models
with those heads galaxy_score and star_score. --save-maps also saves the model's full-image maps, e.g. tidal_map.
"""

import argparse
from pathlib import Path

import numpy as np

from lsst_unet_detection import (BANDS, CONFIG, cutout_bbox, detect_galaxies, detect_in_arrays, detection_maps,
                                 extract_inputs, load_coadd, load_deep_coadds, load_detector, maps_of_arrays,
                                 save_coadd)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="model folder, or its name inside --models-root")
    parser.add_argument("--models-root", type=Path, help="folder holding one sub-folder per trained model")
    parser.add_argument("--out", type=Path, required=True, help="output catalogue (.parquet or .csv)")
    where = parser.add_argument_group("where to look: --tract and --patch, or --ra, --dec and --size, or --coadd")
    where.add_argument("--tract", type=int)
    where.add_argument("--patch", type=int)
    where.add_argument("--ra", type=float, help="centre RA (deg)")
    where.add_argument("--dec", type=float, help="centre Dec (deg)")
    where.add_argument("--size", type=int, default=3072, help="cutout side in pixels (default: %(default)s)")
    where.add_argument("--coadd", type=Path, help="a coadd saved as .npz (e.g. by --save-coadd) instead of the Butler")
    parser.add_argument("--repo", default=CONFIG["butler_repo"], help="Butler repository (default: %(default)s)")
    parser.add_argument("--collections", default=CONFIG["collections"], help="default: %(default)s")
    parser.add_argument("--skymap", default=CONFIG["skymap"], help="default: %(default)s")
    parser.add_argument("--threshold", type=float,
                        help="override the model's calibrated p_detection_centroid threshold")
    parser.add_argument("--save-maps", type=Path,
                        help="also save the model's maps (e.g. tidal_map, spike_map) as an .npz on the image's grid")
    parser.add_argument("--save-coadd", type=Path,
                        help="also save the loaded coadd as an .npz, for scripts/review_detections.py")
    args = parser.parse_args()
    by_patch = args.tract is not None and args.patch is not None
    by_position = args.ra is not None and args.dec is not None
    if by_patch + by_position + (args.coadd is not None) != 1:
        parser.error("give one of: --tract and --patch, --ra and --dec, or --coadd")
    if args.coadd is not None and args.save_coadd is not None:
        parser.error("--save-coadd saves a coadd read from the Butler; --coadd is one already saved")

    detector = load_detector(args.model, args.models_root, args.threshold)
    if args.coadd is not None:
        print(f"Running {detector.name} on {args.coadd}")
        signal, variance, psf_kernels, origin = load_coadd(args.coadd)
        detections = detect_in_arrays(signal, variance, psf_kernels, detector, origin).assign(model=detector.name)
        save_catalogue(detections, args.out, detector, variance)
        if args.save_maps is not None:
            save_maps(maps_of_arrays(signal, variance, psf_kernels, detector), origin, args.save_maps)
        return

    from lsst.daf.butler import Butler  # pyright: ignore[reportMissingImports]  (LSST stack)

    butler = Butler(args.repo, collections=args.collections)
    if by_patch:
        tract, patch, bbox = args.tract, args.patch, None
    else:
        tract, patch, bbox = cutout_bbox(butler, args.ra, args.dec, args.size, args.skymap)
    print(f"Loading {''.join(BANDS)} deep_coadd for tract {tract}, patch {patch}" + (f", {bbox}" if bbox else ""))
    coadds = load_deep_coadds(butler, tract, patch, bbox=bbox, skymap=args.skymap)
    if args.save_coadd is not None:
        args.save_coadd.parent.mkdir(parents=True, exist_ok=True)
        save_coadd(args.save_coadd, coadds)
        print(f"Saved the coadd -> {args.save_coadd}")
    detections = detect_galaxies(coadds, detector).assign(tract=tract, patch=patch, model=detector.name)
    save_catalogue(detections, args.out, detector, extract_inputs(coadds)[1])
    if args.save_maps is not None:
        corner = next(iter(coadds.values())).getBBox()
        save_maps(detection_maps(coadds, detector), [corner.getMinX(), corner.getMinY()], args.save_maps)


def save_catalogue(detections, out, detector, variance):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".parquet":
        detections.to_parquet(out)
    else:
        detections.to_csv(out, index=False)
    print(f"{len(detections):,} detections (bands {detector.band_set(variance)}, p_detection_centroid >= "
          f"{detector.threshold:.4f}) -> {out}")


def save_maps(maps, origin, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **maps, origin=np.array(origin))
    print(f"Saved maps {', '.join(maps)} -> {out}")


if __name__ == "__main__":
    main()
