"""Load a trained U-Net by name and run it over a whole multiband image.

A model folder holds the weights, the input normalisation, the calib peaks (the score -> p_detection_centroid
calibration is refitted from them), the threshold and the model config, which says which heads the model has and how
it fills the area beyond the image edge.

The image is cut into tile_size tiles, each read with a tile_halo border of context; beyond the image edge the border
holds "no data" (zero signal, a huge variance) or, for models made before that, a mirror image. Per band the network
sees arcsinh(S/N / 3) and the normalised log variance, plus the PSF stamps. Detections are the 3x3 local maxima of
the detection heatmap (or, for models without it, the galaxy heatmap) inside each tile, refined by the predicted
sub-pixel offset, calibrated to p_detection_centroid (the probability that the peak is the centre of a real galaxy or
star) and kept if that is at least the model's threshold. Each detection also carries the galaxy and star heatmaps'
values (galaxy_score, star_score) when the model has them.

Detector.maps returns the model's full-image maps (e.g. tidal_map, the probability that each pixel holds detectable
light of a tidal feature), stitched from the tiles.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from sklearn.isotonic import IsotonicRegression

from .config import ARTEFACTS, BANDS, CONFIG, SETTINGS_OF_OLDER_MODELS
from .unet_model import build_unet

SCORE_COLUMN = "p_detection_centroid"


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


def extract_halo(cube, x0, y0, size, halo, fill=None):
    """The tile at (x0, y0) with a halo of context on each side. Beyond the image edge it holds fill, or with
    fill=None a mirror image."""
    ny, nx = cube.shape[1:]
    xs, ys, xe, ye = x0 - halo, y0 - halo, x0 + size + halo, y0 + size + halo
    sx0, sx1, sy0, sy1 = max(0, xs), min(nx, xe), max(0, ys), min(ny, ye)
    patch = np.asarray(cube[:, sy0:sy1, sx0:sx1], np.float32)
    pad = ((0, 0), (sy0 - ys, ye - sy1), (sx0 - xs, xe - sx1))
    if any(p for pair in pad for p in pair):
        if fill is None:
            patch = np.pad(patch, pad, mode="reflect" if min(patch.shape[1:]) > 1 else "edge")
        else:
            patch = np.pad(patch, pad, mode="constant", constant_values=fill)
    return patch


def map_names(model):
    """The model's outputs that are maps of the image (everything but the offset and structure regressions)."""
    return [name for name in model.output_names if name not in ("centroid_offset", "source_structure")]


class Detector:
    """A trained, calibrated U-Net: detect(signal, variance, psf_kernels) -> DataFrame of detections, and
    maps(...) -> its full-image maps."""

    def __init__(self, model, normalisation, calibrator, threshold, cfg, log_re_scaling=None, name=""):
        self.model, self.normalisation, self.calibrator = model, normalisation, calibrator
        self.threshold, self.cfg, self.log_re_scaling, self.name = threshold, cfg, log_re_scaling, name
        outputs = set(model.output_names)
        self.detection_map = "detection_heatmap" if "detection_heatmap" in outputs else "galaxy_heatmap"
        self.kind_maps = [name for name in ("galaxy_heatmap", "star_heatmap")
                          if name in outputs and name != self.detection_map]

    def tiles(self, signal, variance, psf_kernels):
        """Yield (batch of tile corners, predictions) over the whole image."""
        cfg, (ny, nx) = self.cfg, signal.shape[1:]
        size, halo = cfg["tile_size"], cfg["tile_halo"]
        no_data = cfg["edge_padding"] == "no_data"
        n_x, n_y = int(np.ceil(nx / size)), int(np.ceil(ny / size))
        corners = [(tx * size, ty * size) for ty in range(n_y) for tx in range(n_x)]
        for first in range(0, len(corners), cfg["infer_batch"]):
            batch = corners[first:first + cfg["infer_batch"]]
            images = np.stack([encode_planes(extract_halo(signal, x0, y0, size, halo, 0.0 if no_data else None),
                                             extract_halo(variance, x0, y0, size, halo,
                                                          cfg["no_data_variance"] if no_data else None),
                                             self.normalisation)
                               for x0, y0 in batch])
            predictions = self.model.predict_on_batch({"image_planes": images,
                                                       "psf_kernels": np.repeat(psf_kernels[None], len(batch), axis=0)})
            if isinstance(predictions, (list, tuple)):
                predictions = dict(zip(self.model.output_names, predictions))
            yield batch, predictions

    def raw_peaks(self, signal, variance, psf_kernels):
        """Every detection-map peak above min_peak_score, before calibration and thresholding."""
        ny, nx = signal.shape[1:]
        rows = []
        for batch, predictions in self.tiles(signal, variance, psf_kernels):
            for i, (x0, y0) in enumerate(batch):
                rows += self._tile_peaks(predictions, i, x0, y0, nx, ny)
        columns = ["x", "y", "raw_score", "predicted_re_pix",
                   *[name.replace("_heatmap", "_score") for name in self.kind_maps]]
        peaks = pd.DataFrame(rows, columns=columns)
        return peaks[peaks["x"].between(0, nx - 1) & peaks["y"].between(0, ny - 1)].reset_index(drop=True)

    def _tile_peaks(self, predictions, i, x0, y0, nx, ny):
        """Local maxima of one tile's detection map (not its halo), at most max_peaks_per_tile, highest first."""
        size, halo, cap = self.cfg["tile_size"], self.cfg["tile_halo"], self.cfg["max_peaks_per_tile"]
        heatmap = np.asarray(predictions[self.detection_map][i, ..., 0])
        kinds = [np.asarray(predictions[name][i, ..., 0]) for name in self.kind_maps]
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
            rows.append((float(x0 + x - halo + dx), float(y0 + y - halo + dy), float(heatmap[y, x]), re_pix,
                         *[float(kind[y, x]) for kind in kinds]))
        return rows

    def scored_peaks(self, signal, variance, psf_kernels):
        """Every detection-map peak above min_peak_score with its p_detection_centroid, whether or not it passes the
        threshold."""
        peaks = self.raw_peaks(signal, variance, psf_kernels)
        peaks.insert(3, SCORE_COLUMN, self.calibrator.predict(peaks["raw_score"]) if len(peaks) else np.empty(0))
        return peaks

    def detect(self, signal, variance, psf_kernels):
        """Detections with p_detection_centroid >= threshold: x, y (array pixels), raw_score, p_detection_centroid,
        predicted_re_pix (when the model knows its size scaling) and galaxy_score / star_score (when it has those
        heads). signal and variance are (band, y, x) in nJy; psf_kernels is (stamp, stamp, band)."""
        peaks = self.scored_peaks(signal, variance, psf_kernels)
        return peaks.loc[peaks[SCORE_COLUMN].to_numpy(float) >= self.threshold].reset_index(drop=True)

    def maps(self, signal, variance, psf_kernels, names=None):
        """{map name: (ny, nx) float32 image} of the model's maps (default: all of them, e.g. galaxy_heatmap,
        tidal_map, spike_map, detection_heatmap), stitched from the tiles' interiors."""
        names = list(names or map_names(self.model))
        size, halo = self.cfg["tile_size"], self.cfg["tile_halo"]
        ny, nx = signal.shape[1:]
        out = {name: np.zeros((ny, nx), np.float32) for name in names}
        for batch, predictions in self.tiles(signal, variance, psf_kernels):
            for i, (x0, y0) in enumerate(batch):
                h, w = min(size, ny - y0), min(size, nx - x0)
                for name in names:
                    out[name][y0:y0 + h, x0:x0 + w] = np.asarray(predictions[name][i, halo:halo + h, halo:halo + w, 0])
        return out


def load_detector(model, models_root=None, threshold=None):
    """Load a trained model by folder or by name under models_root. threshold overrides the calibrated
    p_detection_centroid cut."""
    model_dir = resolve_model_dir(model, models_root)
    config_path = model_dir / ARTEFACTS["model_config"]
    model_config = json.loads(config_path.read_text()) if config_path.exists() else {}
    cfg = {**CONFIG, **SETTINGS_OF_OLDER_MODELS, **model_config.get("cfg", {})}
    cfg["heads"] = tuple(cfg["heads"])
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
