# lsst_unet_detection

Run a trained U-Net galaxy detector on real LSST `deep_coadd` images. Models come from
[lsst_unet_training](https://github.com/tmsedgwick/lsst_unet_training), which trains them on mocks from
[mock_lsst_image_generation](https://github.com/tmsedgwick/mock_lsst_image_generation).

```python
from lsst.daf.butler import Butler
from lsst_unet_detection import detect_galaxies, load_deep_coadds, load_detector

butler = Butler("dp2", collections="dp2")
coadds = load_deep_coadds(butler, tract=2877, patch=34)            # {band: deep_coadd exposure}, ugrizy
detector = load_detector("mep_unet", models_root="~/unet_models")  # choose the model by name
detections = detect_galaxies(coadds, detector)                     # DataFrame: x, y, ra, dec, p_real, ...
```

## What it does

1. **Get the coadds.** `load_deep_coadds` fetches the `deep_coadd` of each band (ugrizy) for a tract and patch,
   optionally cut to a bounding box. You can also pass exposures you already have, as a `{band: exposure}` dict or a
   `MultibandExposure`.
2. **Extract the three inputs.** For each band: the image and variance arrays (nJy), and a PSF stamp from
   `psf.computeKernelImage()` at the centre of the region, cropped or padded to 25 × 25 and normalised to unit sum,
   as in training.
3. **Neutralise bad pixels.** Pixels with NaN or non-positive variance (NO_DATA, gaps) get zero signal and a huge
   variance, so they read as pure noise.
4. **Run the U-Net.** The image is processed in 256 × 256 tiles with 32 pixels of context. Detections are the peaks
   of the predicted galaxy-centre heatmap, refined by the predicted sub-pixel offset. Raw scores are calibrated to
   `p_real`, the probability a detection is real, and those below the model's threshold (set for 99% purity on the
   mocks) are dropped.
5. **Remove duplicates** within 3 pixels, keeping the highest `p_real`.
6. **Add coordinates**: tract pixel positions and RA/Dec from the coadd WCS.

## Choosing a model

A model is a folder written by lsst_unet_training (weights, input normalisation, calib peaks, threshold, model
config). Keep your models side by side in one folder and pick one by name:

```
~/unet_models/
    mep_unet/
    mep_unet_longer_training/
```

`load_detector("mep_unet", models_root="~/unet_models")` loads `~/unet_models/mep_unet`; you can also pass the
folder path directly. A mistyped name lists the models that are available. Pass `threshold=` to override the
calibrated `p_real` cut, e.g. to trade purity for completeness.

## Output catalogue

| Column | Meaning |
|---|---|
| `x`, `y` | position in the loaded image's pixels (0-based) |
| `x_tract`, `y_tract` | position in tract pixel coordinates |
| `ra`, `dec` | sky position (degrees) |
| `raw_score` | peak height of the predicted galaxy heatmap |
| `p_real` | calibrated probability that the detection is a real galaxy |
| `predicted_re_pix` | predicted half-light radius in pixels (models with a model config) |
| `tract`, `patch`, `model` | added by the command-line script |

## Install

This needs the LSST Science Pipelines for the Butler (`lsst.daf.butler`, `lsst.afw`, `lsst.geom`), so run it where
they are installed, such as a notebook on the Rubin Science Platform. There, clone the repo and install it on top
of the stack:

```bash
git clone https://github.com/tmsedgwick/lsst_unet_detection.git
cd lsst_unet_detection
pip install --user -e . tensorflow scikit-learn
```

For work without the Butler (e.g. running the tests, or on arrays you already have), a plain environment is enough:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example commands

The script needs the LSST stack and Butler access. Defaults are for Data Preview 2 (`--repo dp2 --collections dp2
--skymap lsst_cells_v2`).

```bash
# A whole patch
python scripts/detect_galaxies.py --model mep_unet --models-root ~/unet_models --tract 2877 --patch 34 --out ~/detections/tract2877_patch34.parquet

# A 3072 px (10.2') square centred on a position
python scripts/detect_galaxies.py --model mep_unet --models-root ~/unet_models --ra 59.5 --dec -0.75 --size 3072 --out ~/detections/cirrus.parquet

# The same patch with a different model, to compare
python scripts/detect_galaxies.py --model mep_unet_longer_training --models-root ~/unet_models --tract 2877 --patch 34 --out ~/detections/tract2877_patch34_longer.parquet

# A model folder given directly, and a looser p_real cut (more complete, less pure)
python scripts/detect_galaxies.py --model ~/unet_models/mep_unet --tract 2877 --patch 34 --threshold 0.7 --out ~/detections/loose.csv

# Another Butler repository / collection / skymap
python scripts/detect_galaxies.py --model mep_unet --models-root ~/unet_models --repo /repo/main --collections LSSTCam/runs/DRP/20250501_20250609/w_2025_30/DM-51933 --skymap lsst_cells_v1 --tract 2877 --patch 34 --out ~/detections/main.parquet

# Help
python scripts/detect_galaxies.py --help
```

**From Python, with exposures or arrays you already have**

```python
from lsst_unet_detection import detect_galaxies, extract_inputs, load_detector, neutralise_bad_pixels

detector = load_detector("mep_unet", models_root="~/unet_models")

# deep_coadd exposures you fetched yourself (a {band: exposure} dict or a MultibandExposure)
detections = detect_galaxies(my_coadds, detector)

# or plain arrays: signal and variance (6, H, W) in nJy, psf_kernels (25, 25, 6) unit-sum stamps
signal, variance = neutralise_bad_pixels(signal, variance)
detections = detector.detect(signal, variance, psf_kernels)  # x, y, raw_score, p_real, predicted_re_pix
```

## Tests

```bash
pytest -q
```

The tests use stand-in exposure objects with the afw Exposure interface and a small untrained model, so they run
without the LSST stack. GitHub Actions runs them on every push. On a real DP2 patch (tract 2877, patch 34) the
detector reproduces the original `mep_unet_infer.py` exactly: the same 12,749 detections with identical scores.

## Repository layout

| Path | Contents |
|---|---|
| `lsst_unet_detection/config.py` | settings (`CONFIG`) and model file names |
| `lsst_unet_detection/butler_input.py` | fetching `deep_coadd`s, extracting image / variance / PSF, bad pixels |
| `lsst_unet_detection/unet_model.py` | the network (identical to lsst_unet_training's) |
| `lsst_unet_detection/detector.py` | loading a model by name; tiled inference, peaks and calibration |
| `lsst_unet_detection/pipeline.py` | exposures → catalogue with sky coordinates |
| `scripts/detect_galaxies.py` | command-line entry point |
| `tests/` | pytest suite |

## Licence

MIT (see `LICENSE`).
