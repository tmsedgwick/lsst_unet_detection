"""Default settings for running a trained U-Net on LSST deep coadds.

Network and tiling settings are read from each model's own model config (written by lsst_unet_training); these
defaults are used for any setting a model config does not give.
"""

from typing import Any

BANDS = ["u", "g", "r", "i", "z", "y"]

CONFIG: dict[str, Any] = dict(
    # Tiling and network (must match how the model was trained).
    tile_size=256, tile_halo=32, psf_stamp=25, base_filters=24,
    # Peak finding on the predicted galaxy heatmap.
    min_peak_score=0.03,  # local maxima below this are noise-floor bumps
    max_peaks_per_tile=512,
    infer_batch=16,  # tiles per network call
    duplicate_radius_pix=3.0,  # detections closer than this are merged, keeping the highest p_real
    # Pixels with NaN or non-positive variance (NO_DATA, chip gaps) get zero signal and this multiple of the band's
    # 99.9th-percentile variance, so the network ignores them.
    bad_pixel_variance_factor=1e6,
    # Classical peak finder (peak_finder.py), the comparison for the U-Net: S/N threshold on the PSF-smoothed image,
    # footprint growth in PSF widths, the radius within which peaks from different bands are one detection, and the
    # box size of the local background used to split crowded footprints.
    peak_threshold_sn=5.0, peak_grow_sigmas=2.4, peak_merge_radius_pix=5.0, peak_background_bin_pix=65,
    # Visual review (review.py): U-Net and peak-finder detections closer than this are the same object.
    review_match_radius_pix=3.0,
    # Butler defaults for Rubin Data Preview 2.
    butler_repo="dp2", collections="dp2", skymap="lsst_cells_v2", dataset="deep_coadd",
)

# Files in a model folder, as written by lsst_unet_training.
ARTEFACTS = dict(
    weights="mep_unet_detector.weights.h5",
    normalisation="mep_unet_normalisation.json",
    model_config="mep_unet_model_config.json",  # optional
    calib_peaks="mep_calib_peaks.parquet",
    threshold="mep_threshold.json",
)
