"""Review detections by eye, one at a time in random order, to build a labelled sample of U-Net successes and
failures. Runs on a coadd saved as .npz (e.g. by detect_galaxies.py --save-coadd); the LSST stack is not needed.

    # U-Net detections the classical peak finder did not make
    python scripts/review_detections.py --coadd coadd.npz --model mep_unet --models-root ~/unet_models --unet-only --out feedback_unet_only.json
    # peak-finder detections the U-Net did not make
    python scripts/review_detections.py --coadd coadd.npz --model mep_unet --models-root ~/unet_models --peakfinder-only --out feedback_peakfinder_only.json
    # U-Net peaks just below its threshold (p_real between --min-p-real and the threshold)
    python scripts/review_detections.py --coadd coadd.npz --model mep_unet --models-root ~/unet_models --below-threshold --out feedback_unet_below_threshold.json

Every decision is saved straight away. Run the same command again to continue where you stopped: the candidates and
their order are read back from <out>.candidates.parquet, so the model is not run again. Keys:

    r real   s spurious   u unsure   right / left arrow: next / previous   f: first unreviewed
    + / -: zoom   0: reset zoom   m: markers on / off   shift+click: mark a missed source   x: undo missed   q: quit
    click markers: select them; r / s / u then label the selection and stay put   esc: deselect
"""

import argparse
from pathlib import Path

from lsst_unet_detection import CONFIG, load_detector
from lsst_unet_detection.review import ReviewSession, ReviewWindow, build_candidates, candidates_path, load_coadd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coadd", type=Path, required=True,
                        help=".npz with signal, variance, psf_kernels, bands and optionally origin")
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--unet-only", action="store_const", dest="category", const="unet_only",
                      help="review U-Net detections that the peak finder did not make")
    kind.add_argument("--peakfinder-only", action="store_const", dest="category", const="peakfinder_only",
                      help="review peak-finder detections that the U-Net did not make")
    kind.add_argument("--below-threshold", action="store_const", dest="category", const="unet_below_threshold",
                      help="review U-Net peaks just below the detection threshold")
    parser.add_argument("--out", type=Path, required=True, help="feedback JSON file to write (or continue)")
    parser.add_argument("--model", help="model folder, or its name inside --models-root (needed to start a review)")
    parser.add_argument("--models-root", type=Path, help="folder holding one sub-folder per trained model")
    parser.add_argument("--threshold", type=float, help="override the model's calibrated p_real threshold")
    parser.add_argument("--min-p-real", type=float, default=0.5,
                        help="lowest p_real shown by --below-threshold (default: %(default)s)")
    parser.add_argument("--peak-sn", type=float, default=CONFIG["peak_threshold_sn"],
                        help="S/N threshold of the classical peak finder (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random order (default: %(default)s)")
    args = parser.parse_args()

    signal, variance, psf_kernels, origin = load_coadd(args.coadd)
    details = dict(coadd=str(args.coadd), origin=origin)

    def make_detections():
        if args.model is None:
            parser.error("--model is needed to start a new review")
        detector = load_detector(args.model, args.models_root, args.threshold)
        details.update(model=detector.name, threshold=detector.threshold, min_p_real=args.min_p_real,
                       peak_threshold_sn=args.peak_sn)
        print(f"Running the U-Net ({detector.name}) and the peak finder on {args.coadd.name}...")
        detections = build_candidates(signal, variance, psf_kernels, detector, args.min_p_real,
                                      {**CONFIG, "peak_threshold_sn": args.peak_sn})
        print(detections["category"].value_counts().to_string())
        return detections

    if candidates_path(args.out).exists():
        print(f"Continuing the review in {args.out}")
    try:
        session = ReviewSession.start_or_resume(args.out, args.category, make_detections, args.seed, details)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if len(session.order) == 0:
        print(f"No {args.category} candidates to review.")
        return
    print(f"{len(session.order):,} {args.category} candidates, {session.n_reviewed} reviewed so far. "
          f"Labels are saved to {args.out} after every decision.")

    import matplotlib.pyplot as plt

    window = ReviewWindow(session, signal, variance, psf_kernels)  # noqa: F841  (keeps the key handlers alive)
    plt.show()
    print(f"{session.n_reviewed} of {len(session.order):,} reviewed, {len(session.feedback['missed'])} missed sources "
          f"marked -> {args.out}")


if __name__ == "__main__":
    main()
