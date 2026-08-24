import pickle
from pathlib import Path

import pytest
import torch

from vetime.data.pre_image import ts2image_1d
from vetime.models.vision.legacy_mae.V_encoder import (
    V_model,
    _unfold_image_loop,
    _unfold_image_vectorized,
)


PATCH_SIZE = 14


@pytest.mark.parametrize(
    "size",
    [
        [4, 4, 0, 224, 1],
        [3, 5, 2, 224, 2],
    ],
)
def test_vectorized_unfold_matches_reference_for_uniform_sizes(size):
    """Batching must preserve every sample's original unfold result."""
    torch.manual_seed(64)
    x0 = torch.randn(3, 256, 5)
    sizes = [size] * x0.shape[0]

    expected = _unfold_image_loop(x0, sizes, PATCH_SIZE)
    actual = _unfold_image_vectorized(x0, sizes, PATCH_SIZE)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_vectorized_unfold_matches_reference_for_real_dataset_fold_sizes():
    """Real variable-length samples still share one unfold layout after vectorized folding."""
    dataset_path = Path(__file__).parents[1] / "dataset" / "vetime_train_all_500.pkl"
    if not dataset_path.exists():
        pytest.skip("Local real-data fixture is unavailable")

    with dataset_path.open("rb") as file:
        data = pickle.load(file)
    samples = [sample for sample in data if sample["time_series"].ndim == 1][:3]
    target_length = max(
        ((len(sample["time_series"]) + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
        for sample in samples
    )
    images, padding_values = [], []
    for sample in samples:
        image, _, padding_value = ts2image_1d(sample["time_series"], target_length, PATCH_SIZE)
        images.append(torch.tensor(image, dtype=torch.float32))
        padding_values.append(torch.tensor(padding_value, dtype=torch.float32))

    encoder = V_model.__new__(V_model)
    encoder.patch_size = PATCH_SIZE
    _, sizes = encoder.fold_image_vectorized(torch.stack(images), padding_values)
    x0 = torch.randn(len(samples), 256, 5)

    expected = _unfold_image_loop(x0, sizes, PATCH_SIZE)
    actual = _unfold_image_vectorized(x0, sizes, PATCH_SIZE)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
