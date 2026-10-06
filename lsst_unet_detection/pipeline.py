"""deep_coadd exposures, or the arrays of a saved coadd -> U-Net detection catalogue with tract pixel (and, for
exposures, sky) coordinates, or the model's maps."""

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .butler_input import extract_inputs, neutralise_bad_pixels
from .config import BANDS, CONFIG
from .detector import SCORE_COLUMN


def suppress_duplicates(detections, radius_pix):
    """Keep the highest-scoring (p_detection_centroid) detection within radius_pix of each other (greedy
    non-maximum suppression)."""
    if len(detections) < 2:
        return detections.reset_index(drop=True)
    xy = detections[["x", "y"]].to_numpy(float)
    tree = KDTree(xy)
    suppressed, keep = np.zeros(len(detections), bool), []
    for index in np.argsort(-detections[SCORE_COLUMN].to_numpy(float), kind="stable"):
        if not suppressed[index]:
            keep.append(index)
            suppressed[tree.query_ball_point(xy[index], radius_pix)] = True
    return detections.iloc[sorted(keep)].reset_index(drop=True)


def detect_in_arrays(signal, variance, psf_kernels, detector, origin=(0, 0),
                     duplicate_radius_pix=CONFIG["duplicate_radius_pix"]):
    """Run the detector on (band, y, x) signal and variance with (stamp, stamp, band) PSF stamps, e.g. a coadd saved
    as .npz (load_coadd), and return the catalogue: x, y (pixels within the arrays), x_tract, y_tract (adding origin,
    the tract pixel of the arrays' corner), raw_score, p_detection_centroid, predicted_re_pix and, for models with
    those heads, galaxy_score and star_score. Bad pixels are made "no data" and duplicates suppressed first."""
    signal, variance = neutralise_bad_pixels(signal, variance)
    detections = suppress_duplicates(detector.detect(signal, variance, psf_kernels), duplicate_radius_pix)
    detections.insert(2, "x_tract", detections["x"] + origin[0])
    detections.insert(3, "y_tract", detections["y"] + origin[1])
    return pd.DataFrame(detections)


def maps_of_arrays(signal, variance, psf_kernels, detector, names=None):
    """{map name: (ny, nx) image} of the detector's maps (default: all, e.g. tidal_map) over the arrays."""
    signal, variance = neutralise_bad_pixels(signal, variance)
    return detector.maps(signal, variance, psf_kernels, names)


def detect_galaxies(exposures, detector, bands=BANDS, duplicate_radius_pix=CONFIG["duplicate_radius_pix"]):
    """Run the detector on {band: deep_coadd exposure} (a dict, or a MultibandExposure) and return the catalogue:
    that of detect_in_arrays, with sky positions ra, dec (degrees) added from the exposures' WCS."""
    signal, variance, psf_kernels = extract_inputs(exposures, bands, detector.cfg["psf_stamp"])
    reference = next(exposures[band] for band in bands if band in exposures)  # bands share one pixel grid and WCS
    corner = reference.getBBox()
    detections = detect_in_arrays(signal, variance, psf_kernels, detector, (corner.getMinX(), corner.getMinY()),
                                  duplicate_radius_pix)
    ra, dec = reference.getWcs().pixelToSkyArray(detections["x_tract"].to_numpy(float),
                                                 detections["y_tract"].to_numpy(float), degrees=True)
    detections.insert(4, "ra", np.asarray(ra, float))
    detections.insert(5, "dec", np.asarray(dec, float))
    return detections


def detection_maps(exposures, detector, names=None, bands=BANDS):
    """maps_of_arrays over the exposures, on the exposures' pixel grid."""
    signal, variance, psf_kernels = extract_inputs(exposures, bands, detector.cfg["psf_stamp"])
    return maps_of_arrays(signal, variance, psf_kernels, detector, names)
