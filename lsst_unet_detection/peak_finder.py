"""A classical peak finder that approximates the LSST pipeline's source detection, to compare the U-Net against.

For each band separately:
  1. Smooth the image with a Gaussian of the band's PSF width (a matched filter for point sources).
  2. Estimate the noise as the sigma-clipped scatter of the smoothed image, and mark pixels at S/N >= threshold_sn.
  3. Grow that mask by grow_sigmas PSF widths and label its connected regions: these are the footprints. Growing
     merges nearby blobs into one footprint, as LSST's nSigmaToGrow does.
  4. Peaks are local maxima (3 x 3) of the smoothed image among the above-threshold pixels.
  5. A footprint with several peaks is re-examined after subtracting a coarse local background (medians in
     background_bin_pix boxes), as LSST's temporary local background does: the peaks found then replace the
     footprint's peaks, or if there are none, only its brightest peak is kept.
Peaks from all bands within merge_radius_pix of each other are then merged into one detection, placed at the
brightest of them.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, convolve1d, label, maximum_filter, zoom
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree

from .config import BANDS, CONFIG


def psf_sigma_pix(kernel):
    """Gaussian-equivalent width (sigma, pixels) of a PSF stamp, from its second moment."""
    kernel = np.maximum(np.nan_to_num(np.asarray(kernel, float)), 0.0)
    total, half = kernel.sum(), kernel.shape[0] // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    cx, cy = (kernel * xx).sum() / total, (kernel * yy).sum() / total
    second_moment = (kernel * ((xx - cx) ** 2 + (yy - cy) ** 2)).sum() / total
    return float(np.sqrt(max(second_moment / 2.0, 1e-12)))


def smooth(image, sigma):
    """Image convolved with a unit-sum Gaussian of width sigma pixels (truncated at 4 sigma)."""
    radius = int(4.0 * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    return convolve1d(convolve1d(image, kernel, axis=0, mode="nearest"), kernel, axis=1, mode="nearest")


def clipped_scatter(image):
    """Standard deviation of the image after iterative 3-sigma clipping about the median."""
    values = image[np.isfinite(image)]
    centre = np.median(values)
    for _ in range(5):
        spread = np.std(values)
        keep = np.abs(values - centre) < 3.0 * spread
        if keep.all():
            break
        values = values[keep]
        centre = np.median(values)
    return float(np.std(values))


def coarse_background(image, bin_pix):
    """Median of the image in bin_pix x bin_pix boxes, interpolated back to full resolution."""
    ny, nx = image.shape
    if ny < bin_pix or nx < bin_pix:
        return np.full_like(image, np.median(image))
    n_y, n_x = ny // bin_pix, nx // bin_pix
    medians = np.median(image[:n_y * bin_pix, :n_x * bin_pix].reshape(n_y, bin_pix, n_x, bin_pix), axis=(1, 3))
    return np.asarray(zoom(medians, (ny / n_y, nx / n_x), order=1, mode="nearest"))[:ny, :nx]


def local_maxima(smoothed, sn, mask):
    """Peaks (x, y, brightness, S/N) at the 3 x 3 local maxima of the smoothed image inside mask."""
    y, x = np.nonzero(mask & (smoothed == maximum_filter(smoothed, size=3, mode="nearest")))
    return pd.DataFrame(dict(x=x.astype(float), y=y.astype(float), brightness=smoothed[y, x].astype(float),
                             sn=sn[y, x].astype(float)))


def band_peaks(image, sigma, threshold_sn, grow_sigmas, background_bin_pix):
    """Peaks in one band, following steps 1-5 of the module docstring."""
    smoothed = smooth(image, sigma)
    sn = smoothed / max(clipped_scatter(smoothed), 1e-30)
    above = sn >= threshold_sn
    grow_pix = max(0, int(round(grow_sigmas * sigma)))
    labelled = label(binary_dilation(above, iterations=grow_pix) if grow_pix else above, structure=np.ones((3, 3), int))
    footprints = labelled[0]  # label returns (labels, count)  # pyright: ignore[reportIndexIssue]
    peaks = local_maxima(smoothed, sn, above)
    peaks["footprint"] = footprints[peaks["y"].astype(int), peaks["x"].astype(int)]
    peaks = peaks[peaks["footprint"] > 0]
    peaks_per_footprint = peaks.groupby("footprint")["x"].transform("size")
    if not (peaks_per_footprint > 1).any():
        return peaks

    local = smoothed - coarse_background(smoothed, background_bin_pix)
    local_sn = local / max(clipped_scatter(local), 1e-30)
    local_peaks = local_maxima(local, local_sn, local_sn >= threshold_sn)
    local_peaks["footprint"] = footprints[local_peaks["y"].astype(int), local_peaks["x"].astype(int)]
    kept = [peaks[peaks_per_footprint == 1]]
    for footprint, old in peaks[peaks_per_footprint > 1].groupby("footprint"):
        new = local_peaks[local_peaks["footprint"] == footprint]
        kept.append(new if len(new) else old.nlargest(1, "brightness"))
    return pd.concat(kept, ignore_index=True)


def find_peaks(signal, psf_kernels, bands=BANDS, threshold_sn=CONFIG["peak_threshold_sn"],
               grow_sigmas=CONFIG["peak_grow_sigmas"], merge_radius_pix=CONFIG["peak_merge_radius_pix"],
               background_bin_pix=CONFIG["peak_background_bin_pix"]):
    """Detections of the classical peak finder in a (band, y, x) image with (stamp, stamp, band) PSF stamps.

    Returns one row per detection: x, y (pixels), peak_sn (highest S/N among the merged band peaks), brightness
    (smoothed peak value, nJy) and bands (the bands it was found in, e.g. "g,r,i").
    """
    per_band = []
    for b, band in enumerate(bands):
        image = np.nan_to_num(np.asarray(signal[b], np.float32))
        peaks = band_peaks(image, psf_sigma_pix(psf_kernels[..., b]), threshold_sn, grow_sigmas, background_bin_pix)
        per_band.append(peaks.assign(band=band))
    peaks = pd.concat(per_band, ignore_index=True)
    columns = ["x", "y", "peak_sn", "brightness", "bands"]
    if len(peaks) == 0:
        return pd.DataFrame(columns=columns)

    xy = peaks[["x", "y"]].to_numpy(float)
    pairs = np.array(sorted(KDTree(xy).query_pairs(merge_radius_pix))).reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(peaks),) * 2)
    peaks["group"] = connected_components(graph, directed=False)[1]
    rows = []
    for _, group in peaks.groupby("group"):
        brightest = group.loc[group["brightness"].idxmax()]
        rows.append(dict(x=brightest["x"], y=brightest["y"], peak_sn=float(group["sn"].max()),
                         brightness=float(brightest["brightness"]),
                         bands=",".join(band for band in bands if band in set(group["band"]))))
    return pd.DataFrame(rows, columns=columns)
