"""Checks of the classical peak finder, the U-Net / peak-finder matching, the random-order review session (saving,
resuming) and the review window's keys, on a small synthetic coadd."""

import json
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from lsst_unet_detection import BANDS, find_peaks, load_detector  # noqa: E402
from lsst_unet_detection.review import (ReviewSession, ReviewWindow, build_candidates, candidates_path,  # noqa: E402
                                        load_coadd, match)

SIZE = 200
SOURCES = np.array([[40.0, 50.0], [120.0, 60.0], [70.0, 150.0], [160.0, 160.0]])


def is_source(peaks):
    """Which detections lie within 1 pixel of one of SOURCES."""
    xy = peaks[["x", "y"]].to_numpy(float)
    return (np.hypot(xy[:, None, 0] - SOURCES[None, :, 0], xy[:, None, 1] - SOURCES[None, :, 1]) <= 1).any(axis=1)


def psf_kernels(sigma=1.5):
    yy, xx = np.mgrid[-12:13, -12:13]
    kernel = np.exp(-0.5 * (xx ** 2 + yy ** 2) / sigma ** 2)
    return np.repeat((kernel / kernel.sum())[..., None], len(BANDS), axis=-1).astype(np.float32)


@pytest.fixture(scope="module")
def coadd(tmp_path_factory):
    """A 6-band image with four bright point sources on unit noise, saved as .npz with bands in reverse order."""
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    scene = sum(400 * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / 1.5 ** 2) for x, y in SOURCES)
    signal = np.stack([scene + rng.normal(0, 1, (SIZE, SIZE)) for _ in BANDS]).astype(np.float32)
    variance = np.ones_like(signal)
    path = tmp_path_factory.mktemp("coadd") / "coadd.npz"
    np.savez(path, signal=signal[::-1], variance=variance, psf_kernels=psf_kernels()[..., ::-1],
             bands=np.array(BANDS[::-1]), origin=np.array([1000, 2000]))
    return path


def test_peak_finder_finds_the_sources_and_not_the_noise(coadd):
    signal, _, kernels, origin = load_coadd(coadd)
    assert origin == [1000, 2000] and np.allclose(signal[0], signal[-1], atol=10)
    peaks = find_peaks(signal, kernels)
    sources = is_source(peaks)
    assert sources.sum() == len(SOURCES) and (peaks.loc[sources, "bands"] == ",".join(BANDS)).all()
    # Anything else is an occasional noise peak just over the threshold in a single band, as in the LSST pipeline.
    assert (~peaks.loc[~sources, "bands"].str.contains(",")).all() and peaks.loc[~sources, "peak_sn"].lt(6).all()
    noise_only = np.random.default_rng(5).normal(0, 1, (len(BANDS), SIZE, SIZE)).astype(np.float32)
    assert len(find_peaks(noise_only, kernels)) <= 5


def test_matching():
    unet_matched, peak_matched = match(np.array([[0.0, 0.0], [50.0, 50.0]]), np.array([[2.0, 0.0], [90.0, 90.0]]), 3.0)
    assert unet_matched.tolist() == [True, False] and peak_matched.tolist() == [True, False]


def detections_table(n=30):
    rng = np.random.default_rng(0)
    categories = ["unet_only"] * n + ["peakfinder_only"] * 5 + ["both"] * 5
    return pd.DataFrame(dict(x=rng.uniform(10, 190, len(categories)), y=rng.uniform(10, 190, len(categories)),
                             finder="unet", category=categories, p_real=rng.uniform(0.9, 1, len(categories)),
                             peak_sn=np.nan))


def test_session_random_order_save_and_resume(tmp_path):
    out = tmp_path / "feedback_unet_only.json"
    session = ReviewSession.start_or_resume(out, "unet_only", detections_table, seed=3, details=dict(origin=[10, 20]))
    assert candidates_path(out).exists() and len(session.order) == 30
    assert session.order.tolist() != list(range(30))  # shuffled
    first = session.next_unreviewed()
    session.decide(first, "real")
    session.decide(first + 1, "spurious")
    session.add_missed(5.0, 6.0, first)
    saved = json.loads(out.read_text())
    assert saved["order"] == "random" and saved["n"] == 30 and saved["seed"] == 3
    entry = saved["reviewed"][str(session.order[first])]
    assert entry["decision"] == "real" and entry["x_patch"] == pytest.approx(entry["x"] + 10)
    assert saved["missed"][0]["y_patch"] == 26.0

    def must_not_run():
        raise AssertionError("a resumed review must reuse the saved candidates")

    resumed = ReviewSession.start_or_resume(out, "unet_only", must_not_run, seed=99)
    assert resumed.order.tolist() == session.order.tolist()  # the saved seed wins
    assert resumed.next_unreviewed() == first + 2 and resumed.n_reviewed == 2
    with pytest.raises(ValueError, match="peakfinder_only"):
        ReviewSession.start_or_resume(out, "peakfinder_only", must_not_run)


def test_old_feedback_without_candidates_is_refused(tmp_path):
    out = tmp_path / "feedback.json"
    out.write_text(json.dumps(dict(category="unet_only", n=5, reviewed={}, missed=[])))
    with pytest.raises(FileNotFoundError):
        ReviewSession.start_or_resume(out, "unet_only", detections_table)


def test_window_keys(coadd, tmp_path):
    signal, variance, kernels, _ = load_coadd(coadd)
    session = ReviewSession.start_or_resume(tmp_path / "f.json", "unet_only", detections_table)
    window = ReviewWindow(session, signal, variance, kernels)
    start = window.position
    window.on_key(SimpleNamespace(key="s"))
    assert session.decision(start) == "spurious" and window.position == start + 1
    window.on_key(SimpleNamespace(key="left"))
    assert window.position == start
    window.on_key(SimpleNamespace(key="+"))
    assert window.half_width < window.START_HALF_WIDTH
    window.on_click(SimpleNamespace(inaxes=window.axes[0], xdata=12.0, ydata=13.0, key="shift"))
    window.on_click(SimpleNamespace(inaxes=window.axes[0], xdata=40.0, ydata=40.0, key=None))  # plain click: nothing
    assert len(session.feedback["missed"]) == 1
    window.on_key(SimpleNamespace(key="x"))
    assert session.feedback["missed"] == []
    window.on_key(SimpleNamespace(key="f"))
    assert window.position == session.next_unreviewed()


def test_build_candidates(coadd, models_root):
    signal, variance, kernels, _ = load_coadd(coadd)
    detector = load_detector("tiny", models_root, threshold=0.0)  # untrained: every peak passes
    table = build_candidates(signal, variance, kernels, detector, min_p_real=0.0)
    assert set(table["category"]) <= {"both", "unet_only", "peakfinder_only", "unet_below_threshold"}
    peak_rows = table[table["finder"] == "peakfinder"]
    assert is_source(peak_rows).sum() == len(SOURCES) and peak_rows["peak_sn"].ge(5).all()
    assert table.loc[table["finder"] == "unet", "p_real"].notna().all()
