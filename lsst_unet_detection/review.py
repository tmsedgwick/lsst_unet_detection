"""Visual review of detections, to build a labelled sample of U-Net successes and failures.

The U-Net and the classical peak finder (peak_finder.py) are run on the same coadd and their detections matched
within review_match_radius_pix. Three kinds of candidate can then be reviewed:

  unet_only             U-Net detections (p_real >= threshold) with no peak-finder detection nearby
  peakfinder_only       peak-finder detections with no U-Net detection nearby
  unet_below_threshold  U-Net peaks just below the threshold (min_p_real <= p_real < threshold), whatever the peak
                        finder found; reviewing these tells whether the threshold could be lowered

Candidates are shown in a random order fixed by a seed, so the ones reviewed so far are always a random sample of
the category, however many there are. The labels are saved to a feedback JSON file after every decision:

  {"category": ..., "n": number of candidates, "order": "random", "seed": ..., "coadd": ..., "model": ...,
   "threshold": ..., "origin": [x0, y0],
   "reviewed": {candidate id: {"x", "y", "x_patch", "y_patch", "score", "decision"}, ...},
   "missed": [{"x", "y", "x_patch", "y_patch", "near_candidate"}, ...]}

decision is "real" (a genuine source), "spurious" (an artefact or noise) or "unsure"; "missed" lists sources the
reviewer spotted that no detection caught. x, y are pixels of the coadd array and x_patch, y_patch add its origin.
This is the label format lsst_unet_training's update_threshold_on_aux.py and update_weights_on_aux.py read.

All detections are saved next to the JSON (<name>.candidates.parquet) when a review starts, so reopening it
continues with exactly the same candidates and order, without running the detectors again.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

from .butler_input import neutralise_bad_pixels
from .config import BANDS, CONFIG
from .peak_finder import find_peaks, psf_sigma_pix
from .pipeline import suppress_duplicates

CATEGORIES = ("unet_only", "peakfinder_only", "unet_below_threshold")
DECISIONS = ("real", "spurious", "unsure")


def load_coadd(path, bands=BANDS):
    """(signal, variance, psf_kernels, origin) from an .npz with signal and variance (band, y, x), psf_kernels
    (stamp, stamp, band), bands and optionally origin (the pixel position of the array's corner in the patch)."""
    data = np.load(path)
    stored = [str(band) for band in data["bands"]]
    order = [stored.index(band) for band in bands]
    origin = [int(v) for v in data["origin"]] if "origin" in data.files else [0, 0]
    return (np.asarray(data["signal"], np.float32)[order], np.asarray(data["variance"], np.float32)[order],
            np.asarray(data["psf_kernels"], np.float32)[..., order], origin)


def match(unet_xy, peak_xy, radius):
    """(unet_matched, peak_matched) boolean arrays: a U-Net detection is matched if a peak-finder detection lies
    within radius; that nearest peak-finder detection is then matched too."""
    unet_matched, peak_matched = np.zeros(len(unet_xy), bool), np.zeros(len(peak_xy), bool)
    if len(unet_xy) and len(peak_xy):
        distance, nearest = KDTree(peak_xy).query(unet_xy, k=1)
        unet_matched = np.isfinite(distance) & (distance <= radius)
        peak_matched[nearest[unet_matched]] = True
    return unet_matched, peak_matched


def build_candidates(signal, variance, psf_kernels, detector, min_p_real=0.5, cfg=CONFIG):
    """Every detection of both finders on the coadd, one row each: x, y, finder ("unet" / "peakfinder"), category
    ("both", "unet_only", "peakfinder_only" or "unet_below_threshold"), p_real (U-Net) and peak_sn (peak finder).

    U-Net peaks are merged within duplicate_radius_pix, keeping the highest p_real, before they are split at the
    threshold; below-threshold peaks with p_real < min_p_real are dropped.
    """
    signal, variance = neutralise_bad_pixels(signal, variance, cfg["bad_pixel_variance_factor"])
    unet = suppress_duplicates(detector.scored_peaks(signal, variance, psf_kernels), cfg["duplicate_radius_pix"])
    above = unet["p_real"].to_numpy(float) >= detector.threshold
    below = unet[~above & (unet["p_real"].to_numpy(float) >= min_p_real)]
    unet = unet[above]
    peaks = find_peaks(signal, psf_kernels, threshold_sn=cfg["peak_threshold_sn"],
                       grow_sigmas=cfg["peak_grow_sigmas"], merge_radius_pix=cfg["peak_merge_radius_pix"],
                       background_bin_pix=cfg["peak_background_bin_pix"])
    unet_matched, peak_matched = match(unet[["x", "y"]].to_numpy(float), peaks[["x", "y"]].to_numpy(float),
                                       cfg["review_match_radius_pix"])
    tables = [
        pd.DataFrame(dict(x=unet["x"], y=unet["y"], finder="unet",
                          category=np.where(unet_matched, "both", "unet_only"), p_real=unet["p_real"], peak_sn=np.nan)),
        pd.DataFrame(dict(x=peaks["x"], y=peaks["y"], finder="peakfinder",
                          category=np.where(peak_matched, "both", "peakfinder_only"), p_real=np.nan,
                          peak_sn=peaks["peak_sn"])),
        pd.DataFrame(dict(x=below["x"], y=below["y"], finder="unet", category="unet_below_threshold",
                          p_real=below["p_real"], peak_sn=np.nan)),
    ]
    return pd.concat(tables, ignore_index=True).astype(dict(x=float, y=float, p_real=float, peak_sn=float))


def candidates_path(feedback_path):
    return Path(feedback_path).with_suffix(".candidates.parquet")


class ReviewSession:
    """The candidates of one category in their random order, and the reviewer's labels, saved to feedback_path.

    Candidate ids are row numbers within the category (in the saved candidates table); position p in the review is
    candidate order[p].
    """

    def __init__(self, feedback_path, detections, category, seed=0, details=None):
        if category not in CATEGORIES:
            raise ValueError(f"category {category!r}: choose from {CATEGORIES}")
        self.path, self.detections = Path(feedback_path), detections.reset_index(drop=True)
        self.candidates = self.detections[self.detections["category"] == category].reset_index(drop=True)
        if self.path.exists():
            self.feedback = json.loads(self.path.read_text())
            if self.feedback.get("category") != category or self.feedback.get("order") != "random":
                raise ValueError(f"{self.path} holds a {self.feedback.get('order', 'non-random')}-order review of "
                                 f"{self.feedback.get('category')!r}; choose another --out for a random-order review "
                                 f"of {category!r}")
            seed = int(self.feedback["seed"])
        else:
            self.feedback = dict(category=category, n=int(len(self.candidates)), order="random", seed=int(seed),
                                 **(details or {}), reviewed={}, missed=[])
        self.order = np.random.default_rng(seed).permutation(len(self.candidates))
        self.origin = self.feedback.get("origin", [0, 0])

    @classmethod
    def start_or_resume(cls, feedback_path, category, make_detections, seed=0, details=None):
        """Resume the review saved at feedback_path, or start one with the detections make_detections() returns
        (saving them beside the feedback file so later sessions see the same candidates)."""
        table_path = candidates_path(feedback_path)
        if Path(feedback_path).exists() and not table_path.exists():
            raise FileNotFoundError(f"{feedback_path} exists but {table_path.name} does not, so its candidates cannot "
                                    "be recovered; choose another --out")
        if table_path.exists():
            detections = pd.read_parquet(table_path)
        else:
            detections = make_detections()
            Path(feedback_path).parent.mkdir(parents=True, exist_ok=True)
            detections.to_parquet(table_path)
        session = cls(feedback_path, detections, category, seed, details)
        session.save()
        return session

    def save(self):
        """Write the feedback file (via a temporary file, so an interrupted save cannot corrupt it)."""
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.feedback, indent=2))
        temporary.replace(self.path)

    @property
    def n_reviewed(self):
        return len(self.feedback["reviewed"])

    def candidate(self, position):
        """(candidate id, its row in the candidates table) at a position in the review order."""
        candidate_id = int(self.order[position])
        return candidate_id, self.candidates.iloc[candidate_id]

    def decision(self, position):
        return self.feedback["reviewed"].get(str(self.candidate(position)[0]), {}).get("decision")

    def next_unreviewed(self, start=0):
        """The first position at or after start whose candidate has no decision yet (or None if all are done)."""
        for position in range(start, len(self.order)):
            if self.decision(position) is None:
                return position
        return None

    def decide(self, position, decision):
        if decision not in DECISIONS:
            raise ValueError(f"decision {decision!r}: choose from {DECISIONS}")
        candidate_id, row = self.candidate(position)
        score = row["p_real"] if np.isfinite(row["p_real"]) else row["peak_sn"]
        self.feedback["reviewed"][str(candidate_id)] = dict(
            x=float(row["x"]), y=float(row["y"]), x_patch=float(row["x"]) + self.origin[0],
            y_patch=float(row["y"]) + self.origin[1], score=float(score), decision=decision)
        self.save()

    def add_missed(self, x, y, position):
        self.feedback["missed"].append(dict(x=float(x), y=float(y), x_patch=float(x) + self.origin[0],
                                            y_patch=float(y) + self.origin[1],
                                            near_candidate=self.candidate(position)[0]))
        self.save()

    def undo_missed(self):
        if self.feedback["missed"]:
            self.feedback["missed"].pop()
            self.save()


DISPLAY_PERCENTILES = (1.0, 99.5)  # display range of each panel, from the default-zoom cutout


def colour_image(g, r, i, low, high, softening=10.0):
    """An RGB image (i, r, g as red, green, blue) with a shared arcsinh stretch from low to high (nJy)."""
    stack = np.stack([i, r, g], axis=-1)
    scaled = np.clip((stack - low) / max(float(high - low), 1e-12), 0.0, None)
    return np.clip(np.arcsinh(softening * scaled) / np.arcsinh(softening), 0.0, 1.0)


def snr_images(signal, variance, sigma):
    """(inverse-variance-weighted S/N of all bands combined, the same after a PSF-matched filter of width sigma)."""
    good = np.isfinite(variance) & (variance > 0) & np.isfinite(signal)
    weight = np.where(good, 1.0 / np.where(good, variance, 1.0), 0.0)
    weighted_sum, weight_sum = (np.where(good, signal, 0.0) * weight).sum(0), weight.sum(0)
    stacked = weighted_sum / np.sqrt(np.maximum(weight_sum, 1e-30))
    filtered = gaussian_filter(weighted_sum, sigma, mode="nearest") / np.sqrt(np.maximum(
        gaussian_filter(weight_sum, sigma / np.sqrt(2.0), mode="nearest") / (4.0 * np.pi * sigma ** 2), 1e-30))
    return stacked, filtered


HELP = ("r real   s spurious   u unsure   → / ← next / previous   f first unreviewed   click a marker: review it   "
        "b back   "
        "+ / − zoom   0 reset zoom   m markers on/off   shift+click mark a missed source   x undo missed   q quit")
MARKERS = dict(both=("o", "white"), unet_only=("o", "magenta"), peakfinder_only=("s", "orange"),
               unet_below_threshold=("D", "deepskyblue"))
DECISION_MARKS = dict(real=("✓", "#2ECC40"), spurious=("✗", "#FF4136"), unsure=("~", "#FF851B"))


class ReviewWindow:
    """A matplotlib window showing one candidate at a time: a gri colour cutout, the combined S/N and the
    PSF-matched S/N, with the other detections marked. Keys are listed in HELP."""

    START_HALF_WIDTH = 25  # cutout half-width in pixels

    def __init__(self, session, signal, variance, psf_kernels, bands=BANDS):
        import matplotlib.pyplot as plt

        for key in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[key] = []  # free every key from matplotlib's default shortcuts (s = save, q = quit, ...)
        self.session, self.signal, self.variance = session, signal, variance
        self.colour_bands = [bands.index(band) for band in ("g", "r", "i")]
        self.sigma = float(np.median([psf_sigma_pix(psf_kernels[..., b]) for b in range(len(bands))]))
        self.half_width, self.show_markers = self.START_HALF_WIDTH, True
        self.position = session.next_unreviewed() or 0
        self.return_position = None  # where to go back to (key b) after clicking through to another candidate
        self.position_of = np.argsort(session.order)  # candidate id -> its position in the review order
        self.figure, self.axes = plt.subplots(1, 3, figsize=(14, 5.4))
        self.figure.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.12, wspace=0.03)
        self.figure.text(0.5, 0.02, HELP, ha="center", fontsize=8.5, color="0.3")
        self.add_key()
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.draw()

    def add_key(self):
        """A key to the markers, under the panels."""
        from matplotlib.lines import Line2D

        def symbol(marker, colour, label):
            return Line2D([], [], ls="none", marker=marker, ms=9, mew=1.8, color=colour,
                          markerfacecolor="none", label=label)

        handles = [symbol("+", "cyan", "candidate being reviewed"),
                   symbol(MARKERS["unet_only"][0], MARKERS["unet_only"][1], "U-Net only"),
                   symbol(MARKERS["peakfinder_only"][0], MARKERS["peakfinder_only"][1], "peak finder only"),
                   symbol(MARKERS["both"][0], "0.5", "both (U-Net and peak finder)"),
                   symbol(MARKERS["unet_below_threshold"][0], MARKERS["unet_below_threshold"][1],
                          "U-Net below threshold"),
                   symbol("x", "yellow", "missed source you marked")]
        handles.append(Line2D([], [], ls="none", marker="none",
                              label="✓ / ✗ / ~  reviewed real / spurious / unsure"))
        self.figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=len(handles),
                           fontsize=8, frameon=False, handletextpad=0.3, columnspacing=1.2)

    def cutout(self, cube, cx, cy, half_width=None):
        h = self.half_width if half_width is None else half_width
        x0, y0 = int(round(cx)) - h, int(round(cy)) - h
        out = np.zeros((cube.shape[0], 2 * h + 1, 2 * h + 1), np.float32)
        xa, xb, ya, yb = max(0, x0), min(cube.shape[2], x0 + 2 * h + 1), max(0, y0), min(cube.shape[1], y0 + 2 * h + 1)
        out[:, ya - y0:yb - y0, xa - x0:xb - x0] = cube[:, ya:yb, xa:xb]
        return out, [x0 - 0.5, x0 + 2 * h + 0.5, y0 - 0.5, y0 + 2 * h + 0.5]

    def draw(self):
        session = self.session
        candidate_id, row = session.candidate(self.position)
        cx, cy = float(row["x"]), float(row["y"])
        signal, extent = self.cutout(self.signal, cx, cy)
        variance, _ = self.cutout(self.variance, cx, cy)
        stacked, filtered = snr_images(signal.astype(float), variance.astype(float), self.sigma)
        # The display ranges always come from the default-size cutout, so zooming in or out changes only how much is
        # shown, never the brightness or contrast.
        reference_signal, _ = self.cutout(self.signal, cx, cy, self.START_HALF_WIDTH)
        reference_variance, _ = self.cutout(self.variance, cx, cy, self.START_HALF_WIDTH)
        colour_range = np.nanpercentile(reference_signal[self.colour_bands], DISPLAY_PERCENTILES)
        reference_snr = snr_images(reference_signal.astype(float), reference_variance.astype(float), self.sigma)
        colour = [signal[b] for b in self.colour_bands]
        for ax in self.axes:
            ax.clear()
        self.axes[0].imshow(colour_image(*colour, *colour_range), origin="lower", extent=extent)
        for ax, image, reference, title in [
                (self.axes[1], stacked, reference_snr[0], "S/N, all bands combined"),
                (self.axes[2], filtered, reference_snr[1], f"S/N after PSF-matched filter (σ = {self.sigma:.1f} px)")]:
            low, high = np.nanpercentile(reference, DISPLAY_PERCENTILES)
            ax.imshow(np.arcsinh(np.clip(image, low, high) / 3.0), origin="lower", extent=extent, cmap="gray")
            ax.set_title(title, fontsize=10)
        self.axes[0].set_title("g r i colour", fontsize=10)

        h = self.half_width
        detections = session.detections
        in_view = detections[(np.abs(detections["x"] - cx) <= h) & (np.abs(detections["y"] - cy) <= h)]
        reviewed = {(round(v["x"], 2), round(v["y"], 2)): v["decision"] for v in session.feedback["reviewed"].values()}
        for ax in self.axes:
            if self.show_markers:
                for other in in_view.itertuples():
                    if abs(other.x - cx) < 0.01 and abs(other.y - cy) < 0.01:
                        continue
                    decision = reviewed.get((round(other.x, 2), round(other.y, 2)))
                    if decision:
                        mark, colour = DECISION_MARKS[decision]
                        ax.text(other.x, other.y, mark, color=colour, fontsize=13, fontweight="bold", ha="center",
                                va="center")
                    else:
                        marker, colour = MARKERS[other.category]
                        ax.scatter([other.x], [other.y], s=90, marker=marker, facecolors="none", edgecolors=colour,
                                   linewidths=1.5)
            for missed in session.feedback["missed"]:
                if abs(missed["x"] - cx) <= h and abs(missed["y"] - cy) <= h:
                    ax.plot(missed["x"], missed["y"], "x", color="yellow", ms=12, mew=2)
            ax.plot(cx, cy, "+", color="cyan", ms=16, mew=2)
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
            ax.set_xticks([])
            ax.set_yticks([])

        score = (f"p_real {row['p_real']:.3f}" if np.isfinite(row["p_real"]) else f"peak S/N {row['peak_sn']:.1f}")
        decision = session.decision(self.position) or "not reviewed"
        self.figure.suptitle(
            f"{session.feedback['category']}   ·   candidate {self.position + 1} of {len(session.order)} in random "
            f"order   ·   {score}   ·   {decision}\n{session.n_reviewed} reviewed, {len(session.feedback['missed'])} "
            f"missed sources marked   ·   pixel ({cx:.0f}, {cy:.0f})   ·   saving to {session.path.name}",
            fontsize=10.5)
        self.figure.canvas.draw_idle()

    def go_to(self, position):
        self.position = int(np.clip(position, 0, len(self.session.order) - 1))
        self.draw()

    def on_key(self, event):
        key = event.key or ""
        if key in ("r", "s", "u"):
            self.session.decide(self.position, dict(r="real", s="spurious", u="unsure")[key])
            following = self.session.next_unreviewed(self.position + 1)
            self.go_to(self.position + 1 if following is None else following)
        elif key == "right":
            self.go_to(self.position + 1)
        elif key == "left":
            self.go_to(self.position - 1)
        elif key == "b" and self.return_position is not None:
            self.go_to(self.return_position)
            self.return_position = None
        elif key == "f":
            self.go_to(self.session.next_unreviewed() or 0)
        elif key in ("+", "="):
            self.half_width = max(6, self.half_width // 2)
            self.draw()
        elif key in ("-", "_"):
            self.half_width = min(600, self.half_width * 2)
            self.draw()
        elif key == "0":
            self.half_width = self.START_HALF_WIDTH
            self.draw()
        elif key == "m":
            self.show_markers = not self.show_markers
            self.draw()
        elif key == "x":
            self.session.undo_missed()
            self.draw()
        elif key == "q":
            import matplotlib.pyplot as plt
            plt.close(self.figure)

    def on_click(self, event):
        """Shift+click marks a missed source; a plain click on a marker of a candidate in this review jumps to it."""
        if event.inaxes not in self.axes or event.xdata is None:
            return
        if "shift" in (event.key or ""):
            self.session.add_missed(event.xdata, event.ydata, self.position)
            self.draw()
            return
        candidates = self.session.candidates
        distance = np.hypot(candidates["x"].to_numpy() - event.xdata, candidates["y"].to_numpy() - event.ydata)
        nearest = int(np.argmin(distance)) if len(distance) else None
        if nearest is None or distance[nearest] > max(3.0, self.half_width / 12):
            return  # not on a candidate of this review (other kinds of detection cannot be labelled here)
        position = int(self.position_of[nearest])
        if position != self.position:
            if self.return_position is None:
                self.return_position = self.position
            self.go_to(position)
