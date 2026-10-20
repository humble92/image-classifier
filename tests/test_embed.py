import numpy as np
import pytest
from PIL import Image

from image_classifier.embed import EmbeddingCache, embed_candidates, embed_references
from image_classifier.io_scan import ImageFile, sha256_file


def _file(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(path)
    return ImageFile(path, sha256_file(path))


def test_cache_hits_skip_embedding(tmp_path, fake_embedder):
    files = [_file(tmp_path / f"{i}.png", i) for i in range(5)]
    cache = EmbeddingCache(tmp_path / "cache", fake_embedder.tag)
    v1, err = embed_candidates(fake_embedder, cache, files, batch_size=2)
    assert not err and fake_embedder.calls == 3 and fake_embedder.images_seen == 5
    v2, _ = embed_candidates(fake_embedder, cache, files, batch_size=2)
    assert fake_embedder.calls == 3  # all cached
    for k in v1:
        np.testing.assert_array_equal(v1[k], v2[k])
    assert len(list((tmp_path / "cache" / "fake-v1").glob("*.npy"))) == 5
    assert not list((tmp_path / "cache" / "fake-v1").glob("*.tmp"))


def test_corrupt_cache_entry_recomputed(tmp_path, fake_embedder):
    f = _file(tmp_path / "a.png", 0)
    cache = EmbeddingCache(tmp_path / "cache", fake_embedder.tag)
    embed_candidates(fake_embedder, cache, [f])
    (cache.dir / f"{f.sha256}.npy").write_bytes(b"junk")
    vecs, _ = embed_candidates(fake_embedder, cache, [f])
    assert fake_embedder.calls == 2 and vecs[f.sha256].shape == (192,)


def test_decode_failure_reported(tmp_path, fake_embedder):
    good = _file(tmp_path / "a.png", 0)
    bad_path = tmp_path / "b.png"
    bad_path.write_bytes(b"x")
    bad = ImageFile(bad_path, "deadbeef")
    vecs, errors = embed_candidates(fake_embedder, EmbeddingCache(tmp_path / "c", "t"), [good, bad])
    assert good.sha256 in vecs and "deadbeef" in errors


def test_reference_flip_average(tmp_path, fake_embedder):
    f = _file(tmp_path / "r.png", 3)
    cache = EmbeddingCache(tmp_path / "cache", fake_embedder.tag)
    plain = embed_references(fake_embedder, cache, [f], flip=False)
    flipped = embed_references(fake_embedder, cache, [f], flip=True)
    assert plain.shape == flipped.shape == (1, 192)
    assert np.linalg.norm(flipped[0]) == pytest.approx(1.0, abs=1e-5)
    assert not np.allclose(plain, flipped)
    assert (cache.dir / f"{f.sha256}.flip.npy").exists()


def test_hf_embedder_with_tiny_random_dinov2(tmp_path):
    """Exercises the real HF code path without downloading weights."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from image_classifier.embed import HFEmbedder

    config = transformers.Dinov2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, image_size=56, patch_size=14
    )
    model = transformers.Dinov2Model(config)
    processor = transformers.BitImageProcessor(
        size={"shortest_edge": 64}, crop_size={"height": 56, "width": 56}, do_center_crop=True
    )
    images = [Image.new("RGB", (80, 60), c) for c in [(255, 0, 0), (0, 255, 0), (10, 10, 200)]]
    for patch_mean, dim in [(False, 32), (True, 64)]:
        emb = HFEmbedder(model, processor, "dinov2", "tiny", patch_mean=patch_mean)
        out = emb.embed(images)
        assert out.shape == (3, dim) and out.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert emb.tag == "tiny+patchmean"


def test_hf_embedder_with_tiny_random_siglip():
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from image_classifier.embed import HFEmbedder

    config = transformers.SiglipVisionConfig(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, image_size=32, patch_size=8
    )
    model = transformers.SiglipVisionModel(config)
    processor = transformers.SiglipImageProcessor(size={"height": 32, "width": 32})
    out = HFEmbedder(model, processor, "siglip", "tiny-siglip").embed([Image.new("RGB", (50, 40), (9, 99, 199))] * 2)
    assert out.shape == (2, 32)
    np.testing.assert_allclose(out[0], out[1], atol=1e-5)
