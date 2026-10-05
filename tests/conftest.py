"""Shared fixtures: a models folder holding two small untrained models, "tiny" with the current heads and "original"
with the original layout and a model config that predates the heads and edge settings."""

import json

import keras
import numpy as np
import pandas as pd
import pytest

from lsst_unet_detection import ARTEFACTS, CONFIG
from lsst_unet_detection.config import SETTINGS_OF_OLDER_MODELS
from lsst_unet_detection.unet_model import build_unet


@pytest.fixture(scope="session")
def models_root(tmp_path_factory):
    """A models folder holding one small untrained model, 'tiny', in the lsst_unet_training layout."""
    root = tmp_path_factory.mktemp("models")
    model_dir = root / "tiny"
    model_dir.mkdir()
    cfg = {**CONFIG, "base_filters": 8}
    keras.utils.set_random_seed(0)  # untrained weights, but the same ones whatever order the tests run in
    build_unet(cfg).save_weights(model_dir / ARTEFACTS["weights"])
    normalisation = dict(logvar_centre=[0.0] * 6, logvar_scale=[1.0] * 6)
    (model_dir / ARTEFACTS["normalisation"]).write_text(json.dumps(normalisation))
    (model_dir / ARTEFACTS["model_config"]).write_text(json.dumps(dict(
        cfg=dict(base_filters=8, heads=list(CONFIG["heads"]), edge_padding="no_data"), log_re_mean=0.8,
        log_re_std=0.5)))
    scores = np.linspace(0, 1, 200)
    pd.DataFrame(dict(raw_score=scores, label_real=scores > 0.5)).to_parquet(model_dir / ARTEFACTS["calib_peaks"])
    (model_dir / ARTEFACTS["threshold"]).write_text(json.dumps(dict(threshold=0.5)))

    original = root / "original"
    original.mkdir()
    build_unet({**cfg, **SETTINGS_OF_OLDER_MODELS}).save_weights(original / ARTEFACTS["weights"])
    for artefact in ("normalisation", "calib_peaks", "threshold"):
        (original / ARTEFACTS[artefact]).write_bytes((model_dir / ARTEFACTS[artefact]).read_bytes())
    (original / ARTEFACTS["model_config"]).write_text(json.dumps(dict(cfg=dict(base_filters=8), log_re_mean=0.8,
                                                                      log_re_std=0.5)))
    return root
