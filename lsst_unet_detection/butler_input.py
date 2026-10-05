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
    """{band: deep_coadd exposure} for one patch, optionally cut to bbox (an lsst.geom.Box2I in tract pixels)."""
    parameters = {"bbox": bbox} if bbox is not None else None
    return {band: butler.get(dataset, dict(skymap=skymap, tract=int(tract), patch=int(patch), band=band),
                             parameters=parameters)
            for band in bands}


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
    (stamp, stamp, band) PSF stamps evaluated at the centre of the region."""
    exposures = [exposures[band] for band in bands]
    shapes = {exposure.image.array.shape for exposure in exposures}
    if len(shapes) != 1:
        raise ValueError(f"bands have different shapes: {shapes}")
    signal = np.stack([np.asarray(e.image.array, np.float32) for e in exposures])
    variance = np.stack([np.asarray(e.variance.array, np.float32) for e in exposures])
    psf_kernels = np.stack([psf_stamp(e.getPsf().computeKernelImage(e.getBBox().getCenter()).array, stamp_size)
                            for e in exposures], axis=-1)
    return signal, variance, psf_kernels


def save_coadd(path, exposures, bands=BANDS, stamp_size=CONFIG["psf_stamp"]):
    """Save {band: exposure} as an .npz with signal, variance, psf_kernels, bands and origin (the tract pixel of the
    array's corner), the input of scripts/review_detections.py."""
    signal, variance, psf_kernels = extract_inputs(exposures, bands, stamp_size)
    corner = exposures[bands[0]].getBBox()
    np.savez(path, signal=signal, variance=variance, psf_kernels=psf_kernels, bands=np.array(bands),
             origin=np.array([corner.getMinX(), corner.getMinY()]))


def neutralise_bad_pixels(signal, variance, factor=CONFIG["bad_pixel_variance_factor"]):
    """Copies of signal and variance with NaN / non-positive-variance pixels set to zero signal and a huge variance
    (factor x the band's 99.9th-percentile variance), so they read as pure noise instead of spreading NaNs."""
    signal, variance = signal.copy(), variance.copy()
    for i in range(len(signal)):
        good = np.isfinite(signal[i]) & np.isfinite(variance[i]) & (variance[i] > 0)
        if not good.any():
            raise ValueError(f"band {i} has no valid pixels")
        signal[i][~good] = 0.0
        variance[i][~good] = max(float(np.nanpercentile(variance[i][good], 99.9)) * factor, 1.0)
    return signal, variance
