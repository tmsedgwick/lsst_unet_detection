"""Load a trained U-Net by name and run it over a whole multiband image.

A model folder holds the weights, the input normalisation, the calib peaks (the score -> p_real calibration is
refitted from them) and the p_real threshold; models trained with lsst_unet_training also have a model config.

The image is cut into tile_size tiles, each read with a tile_halo border of context (reflected at the image edge).
Per band the network sees arcsinh(S/N / 3) and the normalised log variance, plus the PSF stamps. Detections are the
3x3 local maxima of the galaxy heatmap inside each tile, refined by the predicted sub-pixel offset, calibrated to
p_real, and kept if p_real is at least the model's threshold.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from sklearn.isotonic import IsotonicRegression

from .config import ARTEFACTS, BANDS, CONFIG
from .unet_model import build_unet


def resolve_model_dir(model, models_root=None):
    """Folder of the named model: model itself if it is a folder, otherwise <models_root>/<model>."""
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    if models_root is not None and (Path(models_root).expanduser() / model).is_dir():
        return Path(models_root).expanduser() / model
    available = (sorted(p.name for p in Path(models_root).expanduser().iterdir() if (p / ARTEFACTS["weights"]).exists())
                 if models_root is not None and Path(models_root).expanduser().is_dir() else [])
    raise FileNotFoundError(f"No model {model!r}" + (f" in {models_root}; available: {available or 'none'}"
                                                     if models_root is not None else ""))


def encode_planes(signal, variance, normalisation):
    """(band, H, W) signal and variance -> (H, W, 2 * n_band) network input: per band arcsinh(S/N / 3) and the
    normalised log variance, both clipped to +-8."""
    channels = []
    for i in range(len(signal)):
        var = np.maximum(np.nan_to_num(variance[i], nan=1e-12), 1e-12)
        sig = np.nan_to_num(signal[i], nan=0.0, posinf=0.0, neginf=0.0)
        snr = np.arcsinh((sig / np.sqrt(var)) / 3.0)
        log_var = (np.log(var) - normalisation["logvar_centre"][i]) / normalisation["logvar_scale"][i]
        channels += [np.clip(snr, -8, 8), np.clip(log_var, -8, 8)]
    return np.stack(channels, axis=-1).astype(np.float32)


def extract_halo(cube, x0, y0, size, halo):
    """The tile at (x0, y0) with a halo of context on each side; beyond the image edge it is filled by reflection."""
    ny, nx = cube.shape[1:]
    xs, ys, xe, ye = x0 - halo, y0 - halo, x0 + size + halo, y0 + size + halo
    sx0, sx1, sy0, sy1 = max(0, xs), min(nx, xe), max(0, ys), min(ny, ye)
    patch = np.asarray(cube[:, sy0:sy1, sx0:sx1], np.float32)
    pad = ((0, 0), (sy0 - ys, ye - sy1), (sx0 - xs, xe - sx1))
    if any(p for pair in pad for p in pair):
        patch = np.pad(patch, pad, mode="reflect" if min(patch.shape[1:]) > 1 else "edge")
    return patch


class Detector:
    """A trained, calibrated U-Net: detect(signal, variance, psf_kernels) -> DataFrame of detections."""

    def __init__(self, model, normalisation, calibrator, threshold, cfg, log_re_scaling=None, name=""):
        self.model, self.normalisation, self.calibrator = model, normalisation, calibrator
        self.threshold, self.cfg, self.log_re_scaling, self.name = threshold, cfg, log_re_scaling, name

    def raw_peaks(self, signal, variance, psf_kernels):
        """Every heatmap peak above min_peak_score, before calibration and thresholding."""
        cfg, (ny, nx) = self.cfg, signal.shape[1:]
        size, halo = cfg["tile_size"], cfg["tile_halo"]
        n_x, n_y = int(np.ceil(nx / size)), int(np.ceil(ny / size))
        tiles = [(tx * size, ty * size) for ty in range(n_y) for tx in range(n_x)]
        rows = []
        for first in range(0, len(tiles), cfg["infer_batch"]):
            batch = tiles[first:first + cfg["infer_batch"]]
            images = np.stack([encode_planes(extract_halo(signal, x0, y0, size, halo),
                                             extract_halo(variance, x0, y0, size, halo), self.normalisation)
                               for x0, y0 in batch])
            predictions = self.model.predict_on_batch({"image_planes": images,
                                                       "psf_kernels": np.repeat(psf_kernels[None], len(batch), axis=0)})
            if isinstance(predictions, (list, tuple)):
                predictions = dict(zip(self.model.output_names, predictions))
            for i, (x0, y0) in enumerate(batch):
                rows += self._tile_peaks(predictions, i, x0, y0, nx, ny)
        peaks = pd.DataFrame(rows, columns=["x", "y", "raw_score", "predicted_re_pix"])
        return peaks[peaks["x"].between(0, nx - 1) & peaks["y"].between(0, ny - 1)].reset_index(drop=True)

    def _tile_peaks(self, predictions, i, x0, y0, nx, ny):
        """Local maxima of one tile's galaxy heatmap (not its halo), at most max_peaks_per_tile, highest first."""
        size, halo, cap = self.cfg["tile_size"], self.cfg["tile_halo"], self.cfg["max_peaks_per_tile"]
        heatmap = np.asarray(predictions["galaxy_heatmap"][i, ..., 0])
        offset = np.asarray(predictions["centroid_offset"][i])
        structure = np.asarray(predictions["source_structure"][i])
        in_tile = np.zeros(heatmap.shape, bool)
        in_tile[halo:halo + min(size, ny - y0), halo:halo + min(size, nx - x0)] = True
        local_max = heatmap == maximum_filter(heatmap, size=3, mode="nearest")
        py, px = np.nonzero(local_max & in_tile & (heatmap >= self.cfg["min_peak_score"]))
        if len(px) > cap:
            keep = np.argpartition(heatmap[py, px], -cap)[-cap:]
            px, py = px[keep], py[keep]
        rows = []
        for x, y in zip(px, py):
            dx, dy = np.clip(offset[y, x], -0.5, 0.5)
            re_pix = np.nan
            if self.log_re_scaling is not None:  # structure head output 0 is scaled log(1 + Re / pixel)
                re_pix = max(float(np.expm1(structure[y, x][0] * self.log_re_scaling[1] + self.log_re_scaling[0])), 0.0)
            rows.append((float(x0 + x - halo + dx), float(y0 + y - halo + dy), float(heatmap[y, x]), re_pix))
        return rows

    def scored_peaks(self, signal, variance, psf_kernels):
        """Every heatmap peak above min_peak_score with its calibrated p_real, whether or not it passes the
        threshold."""
        peaks = self.raw_peaks(signal, variance, psf_kernels)
        peaks["p_real"] = self.calibrator.predict(peaks["raw_score"]) if len(peaks) else np.empty(0)
        return peaks

    def detect(self, signal, variance, psf_kernels):
        """Detections with p_real >= threshold: x, y (array pixels), raw_score, p_real and, when the model knows its
        size scaling, predicted_re_pix. signal and variance are (band, y, x) in nJy; psf_kernels is
        (stamp, stamp, band)."""
        peaks = self.scored_peaks(signal, variance, psf_kernels)
        return peaks.loc[peaks["p_real"].to_numpy(float) >= self.threshold].reset_index(drop=True)


def load_detector(model, models_root=None, threshold=None):
    """Load a trained model by folder or by name under models_root. threshold overrides the calibrated p_real cut."""
    model_dir = resolve_model_dir(model, models_root)
    config_path = model_dir / ARTEFACTS["model_config"]
    model_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    cfg = {**CONFIG, **model_config.get("cfg", {})}
    network = build_unet(cfg)
    network.load_weights(model_dir / ARTEFACTS["weights"])
    normalisation = json.loads((model_dir / ARTEFACTS["normalisation"]).read_text())
    calib_peaks = pd.read_parquet(model_dir / ARTEFACTS["calib_peaks"], columns=["raw_score", "label_real"])
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    calibrator.fit(calib_peaks["raw_score"], calib_peaks["label_real"].astype(int))
    if threshold is None:
        threshold = float(json.loads((model_dir / ARTEFACTS["threshold"]).read_text())["threshold"])
    scaling = (model_config["log_re_mean"], model_config["log_re_std"]) if "log_re_mean" in model_config else None
    if len(normalisation["logvar_centre"]) != len(BANDS):
        raise ValueError(f"model expects {len(normalisation['logvar_centre'])} bands, not {len(BANDS)}")
    return Detector(network, normalisation, calibrator, threshold, cfg, scaling, name=model_dir.name)
