import torch

from dataset.dataloader import collate_fn


def _sample(length, values):
    """Build a minimal real dataset item using the production tuple contract."""
    series = torch.tensor(values, dtype=torch.float32).reshape(length, 1)
    return (
        series,
        torch.zeros(3, 1, length, dtype=torch.float32),
        torch.zeros(length, dtype=torch.long),
        {},
        length,
        torch.zeros(1, 3, dtype=torch.float32),
    )


def test_collate_omits_legacy_masking_and_normal_reference_tensors():
    """Current Query/CMRG training batches must not transport unused legacy tensors."""
    batch = collate_fn(
        [_sample(2, [1.0, 2.0]), _sample(3, [3.0, 4.0, 5.0])],
        patch_size=2,
    )

    assert set(batch) == {
        "time_series",
        "image",
        "labels",
        "attention_mask",
        "period",
        "padding_value",
    }
    assert batch["time_series"].shape == (2, 4, 1)
    assert batch["image"].shape == (2, 3, 1, 4)
