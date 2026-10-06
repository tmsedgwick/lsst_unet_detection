"""Shared fixtures: a models folder holding two small untrained models, "tiny" with the current heads and a band
adapter, calibrated for bands ugrizy and griz, and "centres" with centre heatmaps only (galaxies, star-forming
regions, tidal blobs), mirror-image edges and no adapter, calibrated for ugrizy."""

import json

import keras
import numpy as np
import pandas as pd
import pytest

from lsst_unet_detection import ARTEFACTS, CONFIG
from lsst_unet_detection.unet_model import build_unet


@pytest.fixture(scope="session")
def models_root(tmp_path_factory):
    """A models folder holding the two small untrained models, in the lsst_unet_training layout."""
    root = tmp_path_factory.mktemp("models")
    model_dir = root / "tiny"
    model_dir.mkdir()
    cfg = {**CONFIG, "base_filters": 8, "band_adapter": True}
    keras.utils.set_random_seed(0)  # untrained weights, but the same ones whatever order the tests run in
    build_unet(cfg).save_weights(model_dir / ARTEFACTS["weights"])
    normalisation = dict(logvar_centre=[0.0] * 6, logvar_scale=[1.0] * 6)
    (model_dir / ARTEFACTS["normalisation"]).write_text(json.dumps(normalisation))
    (model_dir / ARTEFACTS["model_config"]).write_text(json.dumps(dict(
        cfg=dict(base_filters=8, heads=list(CONFIG["heads"]), edge_padding="no_data", band_adapter=True,
                 adapter_filters=CONFIG["adapter_filters"]), log_re_mean=0.8, log_re_std=0.5)))
    scores = np.linspace(0, 1, 200)

    def calibrate(folder, band_sets):
        pd.concat([pd.DataFrame(dict(raw_score=scores, label_real=scores > 0.5, band_set=band_set))
                   for band_set in band_sets]).to_parquet(folder / ARTEFACTS["calib_peaks"])
        (folder / ARTEFACTS["threshold"]).write_text(json.dumps(dict(
            band_sets={band_set: dict(threshold=0.5) for band_set in band_sets})))

    calibrate(model_dir, ["ugrizy", "griz"])

    centres = root / "centres"
    centres.mkdir()
    heads = ["galaxy_heatmap", "sfregion_heatmap", "tidal_heatmap"]
    build_unet({**cfg, "heads": heads, "band_adapter": False}).save_weights(centres / ARTEFACTS["weights"])
    (centres / ARTEFACTS["normalisation"]).write_text(json.dumps(normalisation))
    calibrate(centres, ["ugrizy"])
    (centres / ARTEFACTS["model_config"]).write_text(json.dumps(dict(
        cfg=dict(base_filters=8, heads=heads, edge_padding="reflect", band_adapter=False), log_re_mean=0.8,
        log_re_std=0.5)))
    return root
