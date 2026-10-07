"""download_model, against a release zip served from a local file:// URL."""

import zipfile

import pytest

from lsst_unet_detection import ARTEFACTS, load_detector
from lsst_unet_detection.published_models import PUBLISHED_MODELS, download_model, sha256_of


def release_zip(models_root, name, out_dir):
    """A zip of the fixture model "tiny" laid out as a published model called name, and its SHA-256."""
    archive = out_dir / f"{name}.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for file in ARTEFACTS.values():
            bundle.write(models_root / "tiny" / file, f"{name}/{file}")
    return archive, sha256_of(archive)


def test_download_installs_a_loadable_model(models_root, tmp_path):
    archive, digest = release_zip(models_root, "published", tmp_path)
    published = {"published": dict(url=archive.as_uri(), sha256=digest, about="test")}
    target = download_model("published", tmp_path / "models", published)
    assert target == tmp_path / "models" / "published"
    assert all((target / file).exists() for file in ARTEFACTS.values())
    assert load_detector("published", models_root=tmp_path / "models").name == "published"
    assert download_model("published", tmp_path / "models", published) == target  # already there: left as it is


def test_download_refuses_a_corrupted_file(models_root, tmp_path):
    archive, _ = release_zip(models_root, "published", tmp_path)
    published = {"published": dict(url=archive.as_uri(), sha256="0" * 64, about="test")}
    with pytest.raises(ValueError, match="SHA-256"):
        download_model("published", tmp_path / "models", published)
    assert not (tmp_path / "models" / "published").exists()


def test_download_refuses_an_incomplete_model(models_root, tmp_path):
    archive = tmp_path / "published.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(models_root / "tiny" / ARTEFACTS["weights"], f"published/{ARTEFACTS['weights']}")
    published = {"published": dict(url=archive.as_uri(), sha256=sha256_of(archive), about="test")}
    with pytest.raises(ValueError, match="lacks"):
        download_model("published", tmp_path / "models", published)
    assert not (tmp_path / "models" / "published").exists()


def test_unknown_model_names_the_published_ones(tmp_path):
    with pytest.raises(KeyError, match="mep_unet_v1"):
        download_model("no_such_model", tmp_path)


def test_published_models_point_at_this_repository():
    for entry in PUBLISHED_MODELS.values():
        assert entry["url"].startswith("https://github.com/tmsedgwick/lsst_unet_detection/releases/download/")
        assert len(entry["sha256"]) == 64

