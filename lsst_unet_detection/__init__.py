"""Run a trained U-Net galaxy detector on LSST deep_coadd images."""

from .butler_input import cutout_bbox, extract_inputs, load_deep_coadds, neutralise_bad_pixels
from .config import ARTEFACTS, BANDS, CONFIG
from .detector import Detector, load_detector
from .pipeline import detect_galaxies

__all__ = ["ARTEFACTS", "BANDS", "CONFIG", "Detector", "cutout_bbox", "detect_galaxies", "extract_inputs",
           "load_deep_coadds", "load_detector", "neutralise_bad_pixels"]
