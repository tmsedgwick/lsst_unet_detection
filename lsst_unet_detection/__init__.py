"""Run a trained U-Net galaxy detector on LSST deep_coadd images."""

from .butler_input import (cutout_bbox, extract_inputs, load_coadd, load_deep_coadds, neutralise_bad_pixels,
                           save_coadd)
from .config import ARTEFACTS, BANDS, CONFIG
from .detector import Detector, load_detector
from .peak_finder import find_peaks
from .pipeline import detect_galaxies, detect_in_arrays, detection_maps, maps_of_arrays
from .published_models import PUBLISHED_MODELS, download_model

__all__ = ["ARTEFACTS", "BANDS", "CONFIG", "PUBLISHED_MODELS", "Detector", "cutout_bbox", "detect_galaxies",
           "detect_in_arrays", "detection_maps", "download_model", "extract_inputs", "find_peaks", "load_coadd",
           "load_deep_coadds", "load_detector", "maps_of_arrays", "neutralise_bad_pixels", "save_coadd"]
