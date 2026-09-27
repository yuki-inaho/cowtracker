"""Sparse tracking of query pixels: pixel-centre size mapping, bilinear sampling, job validation, one forward pass."""

import copy

import numpy as np
import pytest
from fakes import TranslatingTracker
from PIL import Image

from cowtracker.toolkit.sparse_tracking import (
    JOBS_SCHEMA,
    bilinear_sample,
    rescale_pixels,
    track_queries,
    validate_jobs,
)

SOURCE_HW, INFERENCE_HW = (288, 384), (336, 448)
WINDOW = 4


def test_mapping_roundtrip():
    rng = np.random.default_rng(0)
    xy = rng.uniform([0, 0], [SOURCE_HW[1] - 1, SOURCE_HW[0] - 1], size=(1000, 2))

    back = rescale_pixels(rescale_pixels(xy, SOURCE_HW, INFERENCE_HW), INFERENCE_HW, SOURCE_HW)

    np.testing.assert_allclose(back, xy, rtol=0, atol=1e-5)
    # u_inf = (u + 0.5) * W_inf / W_src - 0.5 (pixel centres, as cv2.INTER_LINEAR resizes)
    np.testing.assert_allclose(rescale_pixels([[0.0, 0.0]], SOURCE_HW, INFERENCE_HW), [[1 / 12, 1 / 12]])
    np.testing.assert_allclose(
        rescale_pixels([[383.0, 287.0]], SOURCE_HW, INFERENCE_HW), [[447 - 1 / 12, 335 - 1 / 12]]
    )


def test_bilinear_sample_linear_field():
    frames, height, width = 3, 7, 9
    t, y, x = np.meshgrid(np.arange(frames), np.arange(height), np.arange(width), indexing="ij")
    field = np.stack([2.0 * x + 3.0 * y + 5.0 * t, -x + 0.5 * y], axis=-1)  # [T, H, W, 2]
    xy = np.array([[0.0, 0.0], [width - 1, height - 1], [3.25, 1.5], [7.9, 5.1], [0.5, 6.0]])

    sampled = bilinear_sample(field, xy)
    scalar = bilinear_sample(field[..., 0], xy)

    ts = np.arange(frames)[None]
    expected = np.stack(
        [2.0 * xy[:, :1] + 3.0 * xy[:, 1:] + 5.0 * ts, np.broadcast_to(-xy[:, :1] + 0.5 * xy[:, 1:], (5, 3))], -1
    )
    assert sampled.shape == (5, frames, 2)
    np.testing.assert_allclose(sampled, expected, atol=1e-9)
    np.testing.assert_allclose(scalar, expected[..., 0], atol=1e-9)


def test_bilinear_sample_reads_the_edge_pixel_within_half_a_pixel_and_rejects_positions_off_the_image():
    field = np.arange(12.0).reshape(1, 3, 4)

    np.testing.assert_allclose(bilinear_sample(field, [[-0.5, 0.0], [3.5, 2.5]]), [[0.0], [11.0]])
    with pytest.raises(ValueError, match="off the"):
        bilinear_sample(field, [[-0.6, 0.0]])


def test_identity_tracker_returns_queries():
    queries = np.array([[0.0, 0.0], [383.0, 287.0], [100.25, 57.75], [191.5, 143.5]])
    tracker = TranslatingTracker(step=(0.0, 0.0))

    uv, visconf = track_queries(tracker, np.zeros((5, *INFERENCE_HW, 3), np.uint8), queries, SOURCE_HW)

    assert tracker.lengths == [5]  # one forward pass
    assert uv.shape == (4, 5, 2) and uv.dtype == np.float32
    assert visconf.shape == (4, 5) and visconf.dtype == np.float32
    np.testing.assert_allclose(uv, np.broadcast_to(queries[:, None], uv.shape), atol=1e-4)
    np.testing.assert_allclose(visconf, 0.5)


def test_translating_tracker_shift_is_returned_in_source_pixels():
    queries = np.array([[10.0, 20.0], [200.5, 100.25]])
    step = np.array([1.0, 2.0])  # inference px per frame

    uv, _ = track_queries(TranslatingTracker(step=step), np.zeros((4, *INFERENCE_HW, 3), np.uint8), queries, SOURCE_HW)

    source_step = step * np.array([SOURCE_HW[1] / INFERENCE_HW[1], SOURCE_HW[0] / INFERENCE_HW[0]])
    expected = queries[:, None] + np.arange(4)[None, :, None] * source_step
    np.testing.assert_allclose(uv, expected, atol=1e-4)


def _spec(tmp_path, count=5, size_hw=(60, 80)):
    """A valid job over the first 3 of ``count`` images in ``tmp_path / "rgb"``."""
    directory = tmp_path / "rgb"
    directory.mkdir()
    for index in range(count):
        Image.fromarray(np.full((*size_hw, 3), 30 * index, np.uint8)).save(directory / f"frame_{index:06d}.png")
    job = {
        "id": "dataset_000_s000000_r000001",
        "image_dir": str(directory),
        "frames": [f"frame_{index:06d}.png" for index in range(1, 4)],
        "queries": [[0.0, 0.0], [79.0, 59.0], [12.5, 30.25]],
    }
    return {"schema": JOBS_SCHEMA, "jobs": [job]}


def test_validate_jobs_reads_sizes_paths_and_queries(tmp_path):
    (job,) = validate_jobs(_spec(tmp_path), window_len=WINDOW)

    assert job.id == "dataset_000_s000000_r000001"
    assert job.source_hw == (60, 80)
    assert [path.name for path in job.paths] == ["frame_000001.png", "frame_000002.png", "frame_000003.png"]
    np.testing.assert_array_equal(job.queries, [[0.0, 0.0], [79.0, 59.0], [12.5, 30.25]])


def _duplicate(spec, job):
    spec["jobs"].append(copy.deepcopy(job))


def _resized_frame(spec, job):
    Image.fromarray(np.zeros((70, 80, 3), np.uint8)).save(f"{job['image_dir']}/frame_000004.png")
    job["frames"][-1] = "frame_000004.png"


INVALID = {
    "wrong schema": (lambda spec, job: spec.update(schema="cow_sparse_jobs_v0"), "schema"),
    "no jobs": (lambda spec, job: spec.update(jobs=[]), "no jobs"),
    "missing key": (lambda spec, job: job.pop("queries"), "keys"),
    "unexpected key": (lambda spec, job: job.update(split=0), "keys"),
    "id not a file name": (lambda spec, job: job.update(id="../escape"), "file name"),
    "relative image dir": (lambda spec, job: job.update(image_dir="rgb"), "absolute"),
    "missing image": (lambda spec, job: job["frames"].__setitem__(1, "frame_000099.png"), "missing image"),
    "one frame": (lambda spec, job: job.update(frames=job["frames"][:1]), "frames"),
    "frames over window": (lambda spec, job: job.update(frames=[f"frame_{i:06d}.png" for i in range(5)]), "window"),
    "no queries": (lambda spec, job: job.update(queries=[]), "queries"),
    "query not a pair": (lambda spec, job: job.update(queries=[[1.0, 2.0, 3.0]]), "queries"),
    "query outside the image": (lambda spec, job: job["queries"].append([80.0, 10.0]), "outside"),
    "nan query": (lambda spec, job: job["queries"].append([float("nan"), 10.0]), "outside"),
    "duplicate id": (_duplicate, "duplicate"),
    "mismatched image sizes": (_resized_frame, "size"),
}


@pytest.mark.parametrize("case", INVALID)
def test_job_validation_errors(tmp_path, case):
    mutate, message = INVALID[case]
    spec = _spec(tmp_path)
    mutate(spec, spec["jobs"][0])

    with pytest.raises(ValueError, match=message):
        validate_jobs(spec, window_len=WINDOW)
