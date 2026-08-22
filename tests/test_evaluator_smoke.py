import torch
from torch import nn

from vetime.engine.evaluator import Evaluator


def test_evaluator_returns_mean_loss_without_updating_parameters():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    before = model.weight.detach().clone()
    evaluator = Evaluator()

    def step_fn(current_model, batch):
        features, target = batch
        return (current_model(features) - target).square().mean()

    result = evaluator.evaluate_validation(
        model,
        [(torch.tensor([[1.0]]), torch.tensor([[2.0]]))],
        step_fn,
    )

    assert result.loss == 1.0
    torch.testing.assert_close(model.weight, before)
