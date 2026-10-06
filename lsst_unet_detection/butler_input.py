"""Get deep_coadd exposures from the Butler and extract the three inputs the U-Net needs.

For each band the detector needs the image and variance arrays (nJy) and a PSF stamp. The PSF is evaluated with
psf.computeKernelImage() at the centre of the region, then centre-cropped or zero-padded to psf_stamp x psf_stamp and
normalised to unit sum, which is how the network saw PSFs in training.

Only load_deep_coadds and cutout_bbox import the LSST stack; the extraction works on any object with the afw
Exposure interface (image, variance, getPsf, getBBox), e.g. an ExposureF or one band of a MultibandExposure.
"""

import numpy as np

from .config import BANDS, CONFIG


def load_deep_coadds(butler, tract, patch, bands=BANDS, bbox=None, skymap=CONFIG["skymap"],
                     dataset=CONFIG["dataset"]):
    """{band: deep_coadd exposure} for one patch, optionally cut to bbox (an lsst.geom.Box2I in tract pixels). Bands
    without a deep_coadd are left out (the detector then treats them as missing)."""
    parameters = {"bbox": bbox} if bbox is not None else None
    exposures = {}
    for band in bands:
        try:
            exposures[band] = butler.get(dataset, dict(skymap=skymap, tract=int(tract), patch=int(patch), band=band),
                                         parameters=parameters)
        except LookupError:  # the Butler's DatasetNotFoundError
            print(f"No {dataset} in band {band} for tract {tract}, patch {patch}: treated as missing")
    if not exposures:
        raise LookupError(f"no {dataset} in any of the bands {bands} for tract {tract}, patch {patch}")
    return exposures


def cutout_bbox(butler, ra_deg, dec_deg, size_pix, skymap=CONFIG["skymap"]):
    """(tract, patch, bbox) of a size_pix square centred as close to (RA, Dec) as the patch allows. The side is
    rounded down to a whole number of 256 px U-Net tiles and kept inside the patch."""
    import lsst.geom as geom  # pyright: ignore[reportMissingImports]  (LSST stack)

    sky_map = butler.get("skyMap", skymap=skymap)
    coord = geom.SpherePoint(ra_deg, dec_deg, geom.degrees)
    tract_info = sky_map.findTract(coord)
    patch_info = tract_info.findPatch(coord)
    patch_box = patch_info.getOuterBBox()
    side = min(int(size_pix), patch_box.getWidth(), patch_box.getHeight())
    side = max(256, side // 256 * 256)
    centre = tract_info.getWcs().skyToPixel(coord)

    def start(centre_pix, low, high):  # centred on the target, but kept inside [low, high]
        return min(max(int(np.rint(centre_pix - (side - 1) / 2.0)), low), high - side + 1)

    corner = geom.Point2I(start(centre.x, patch_box.getMinX(), patch_box.getMaxX()),
                          start(centre.y, patch_box.getMinY(), patch_box.getMaxY()))
    return int(tract_info.getId()), int(tract_info.getSequentialPatchIndex(patch_info)), \
        geom.Box2I(corner, geom.Extent2I(side, side))


def psf_stamp(kernel, size=CONFIG["psf_stamp"]):
    """A PSF kernel image centre-cropped or zero-padded to size x size and normalised to unit sum."""
    kernel = np.nan_to_num(np.asarray(kernel, np.float32))
    (ky, kx), half = kernel.shape, size // 2
    cy, cx = ky // 2, kx // 2
    ys, ye, xs, xe = max(cy - half, 0), min(cy + half + 1, ky), max(cx - half, 0), min(cx + half + 1, kx)
    stamp = np.zeros((size, size), np.float32)
    stamp[half - (cy - ys):half + (ye - cy), half - (cx - xs):half + (xe - cx)] = kernel[ys:ye, xs:xe]
    total = float(stamp.sum())
    if total <= 0:
        raise ValueError("PSF kernel has no positive flux")
    return stamp / total


def extract_inputs(exposures, bands=BANDS, stamp_size=CONFIG["psf_stamp"]):
    """(signal, variance, psf_kernels) from {band: exposure}: (band, y, x) float32 image and variance arrays and
    (stamp, stamp, band) PSF stamps evaluated at the centre of the region. A band missing from exposures gets "no
    data" (zero signal, variance no_data_variance) and the mean of the other bands' PSF stamps."""
    present = [band for band in bands if band in exposures]
    if not present:
        raise ValueError("no exposures in any band")
    shapes = {exposures[band].image.array.shape for band in present}
    if len(shapes) != 1:
        raise ValueError(f"bands have different shapes: {shapes}")
    shape = shapes.pop()
    stamps = {band: psf_stamp(exposures[band].getPsf().computeKernelImage(exposures[band].getBBox().getCenter()).array,
                              stamp_size) for band in present}
    mean_stamp = np.mean(list(stamps.values()), axis=0)
    signal = np.stack([np.asarray(exposures[band].image.array, np.float32) if band in exposures
                       else np.zeros(shape, np.float32) for band in bands])
    variance = np.stack([np.asarray(exposures[band].variance.array, np.float32) if band in exposures
                         else np.full(shape, CONFIG["no_data_variance"], np.float32) for band in bands])
    psf_kernels = np.stack([stamps.get(band, mean_stamp / mean_stamp.sum()) for band in bands], axis=-1)
    return signal, variance, psf_kernels


def save_coadd(path, exposures, bands=BANDS, stamp_size=CONFIG["psf_stamp"]):
    """Save {band: exposure} as an .npz with signal, variance, psf_kernels, bands and origin (the tract pixel of the
    array's corner), the input of scripts/review_detections.py."""
    signal, variance, psf_kernels = extract_inputs(exposures, bands, stamp_size)
    corner = next(exposures[band] for band in bands if band in exposures).getBBox()
    np.savez(path, signal=signal, variance=variance, psf_kernels=psf_kernels, bands=np.array(bands),
             origin=np.array([corner.getMinX(), corner.getMinY()]))


def neutralise_bad_pixels(signal, variance, no_data_variance=CONFIG["no_data_variance"]):
    """Copies of signal and variance with NaN / non-positive-variance pixels (NO_DATA, chip gaps) set to "no data":
    zero signal and variance no_data_variance, as the network was trained to read missing data. A band with no valid
    pixels at all is then simply a missing band."""
    signal, variance = signal.copy(), variance.copy()
    bad = ~(np.isfinite(signal) & np.isfinite(variance) & (variance > 0))
    signal[bad], variance[bad] = 0.0, no_data_variance
    return signal, variance
