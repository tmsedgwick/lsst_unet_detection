"""Checks of PSF stamps, bad-pixel handling, model lookup and the full exposure -> catalogue path, using stand-in
objects with the afw Exposure interface (so the LSST stack is not needed) and a small untrained model."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lsst_unet_detection import ARTEFACTS, BANDS, CONFIG, detect_galaxies, extract_inputs, load_detector
from lsst_unet_detection.butler_input import neutralise_bad_pixels, psf_stamp
from lsst_unet_detection.pipeline import suppress_duplicates
from lsst_unet_detection.unet_model import build_unet

SIZE, CORNER = 300, (1000, 2000)  # image side and its tract-pixel corner


class FakeBBox:
    def getMinX(self):
        return CORNER[0]

    def getMinY(self):
        return CORNER[1]

    def getCenter(self):
        return (CORNER[0] + SIZE / 2, CORNER[1] + SIZE / 2)


class FakeWcs:
    def pixelToSkyArray(self, x, y, degrees=True):
        return 150.0 + np.asarray(x) * 0.2 / 3600, 2.0 + np.asarray(y) * 0.2 / 3600


class FakeExposure:
    """image / variance / getPsf / getBBox / getWcs, like an lsst.afw.image.ExposureF."""

    def __init__(self, image, variance, psf_sigma_pix):
        self.image, self.variance = SimpleNamespace(array=image), SimpleNamespace(array=variance)
        yy, xx = np.mgrid[-20:21, -20:21]
        kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / psf_sigma_pix ** 2)
        self._psf = SimpleNamespace(computeKernelImage=lambda point: SimpleNamespace(array=kernel / kernel.sum()))

    def getPsf(self):
        return self._psf

    def getBBox(self):
        return FakeBBox()

    def getWcs(self):
        return FakeWcs()


@pytest.fixture(scope="module")
def exposures():
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    scene = sum(500 * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / 4.0) for x, y in rng.uniform(20, 280, (20, 2)))
    return {band: FakeExposure((scene + rng.normal(0, 1, (SIZE, SIZE))).astype(np.float32),
                               np.ones((SIZE, SIZE), np.float32), 2.2) for band in BANDS}


@pytest.fixture(scope="module")
def models_root(tmp_path_factory):
    """A models folder holding one small untrained model, 'tiny', in the lsst_unet_training layout."""
    root = tmp_path_factory.mktemp("models")
    model_dir = root / "tiny"
    model_dir.mkdir()
    cfg = {**CONFIG, "base_filters": 8}
    build_unet(cfg).save_weights(model_dir / ARTEFACTS["weights"])
    normalisation = dict(logvar_centre=[0.0] * 6, logvar_scale=[1.0] * 6)
    (model_dir / ARTEFACTS["normalisation"]).write_text(json.dumps(normalisation))
    (model_dir / ARTEFACTS["model_config"]).write_text(json.dumps(dict(cfg=dict(base_filters=8), log_re_mean=0.8,
                                                                       log_re_std=0.5)))
    scores = np.linspace(0, 1, 200)
    pd.DataFrame(dict(raw_score=scores, label_real=scores > 0.5)).to_parquet(model_dir / ARTEFACTS["calib_peaks"])
    (model_dir / ARTEFACTS["threshold"]).write_text(json.dumps(dict(threshold=0.5)))
    return root


def test_psf_stamp_is_centred_unit_sum():
    yy, xx = np.mgrid[-20:21, -20:21]
    stamp = psf_stamp(np.exp(-0.5 * (xx ** 2 + yy ** 2) / 4.0))  # 41 x 41 -> cropped
    assert stamp.shape == (25, 25) and stamp.sum() == pytest.approx(1.0) and stamp.argmax() == 12 * 25 + 12
    assert psf_stamp(np.ones((5, 5))).sum() == pytest.approx(1.0)  # 5 x 5 -> zero-padded
    with pytest.raises(ValueError):
        psf_stamp(np.zeros((25, 25)))


def test_bad_pixels_become_noise():
    signal, variance = np.ones((6, 10, 10), np.float32), np.ones((6, 10, 10), np.float32)
    signal[0, 0, 0], variance[1, 1, 1] = np.nan, 0.0
    clean_signal, clean_variance = neutralise_bad_pixels(signal, variance)
    assert clean_signal[0, 0, 0] == 0 and clean_variance[0, 0, 0] == 1e6 and clean_variance[1, 1, 1] == 1e6
    assert np.isnan(signal[0, 0, 0])  # inputs untouched


def test_extract_inputs(exposures):
    signal, variance, psf_kernels = extract_inputs(exposures)
    assert signal.shape == variance.shape == (6, SIZE, SIZE) and psf_kernels.shape == (25, 25, 6)
    assert np.allclose(psf_kernels.sum(axis=(0, 1)), 1.0)


def test_model_lookup_by_name(models_root):
    assert load_detector("tiny", models_root).name == "tiny"
    assert load_detector(str(models_root / "tiny")).threshold == 0.5
    assert load_detector("tiny", models_root, threshold=0.9).threshold == 0.9
    with pytest.raises(FileNotFoundError, match="available: \\['tiny'\\]"):
        load_detector("missing", models_root)


def test_suppress_duplicates_keeps_best():
    detections = pd.DataFrame(dict(x=[10.0, 11.0, 50.0], y=[10.0, 10.5, 50.0], p_real=[0.6, 0.9, 0.7]))
    assert suppress_duplicates(detections, 3.0)["p_real"].tolist() == [0.9, 0.7]


def test_detect_galaxies_catalogue(exposures, models_root):
    detector = load_detector("tiny", models_root, threshold=0.0)  # untrained, so keep everything
    catalogue = detect_galaxies(exposures, detector)
    assert list(catalogue.columns[:6]) == ["x", "y", "x_tract", "y_tract", "ra", "dec"]
    assert len(catalogue) > 0
    assert np.allclose(catalogue["x_tract"] - catalogue["x"], CORNER[0])
    assert np.allclose(catalogue["dec"], 2.0 + catalogue["y_tract"] * 0.2 / 3600)
    assert bool(catalogue["predicted_re_pix"].notna().all())
