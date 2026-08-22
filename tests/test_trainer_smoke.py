import torch
from torch import nn

from vetime.engine.trainer import Trainer, TrainerDependencies


def test_trainer_updates_a_trainable_parameter_on_one_batch():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = (torch.tensor([[1.0]]), torch.tensor([[2.0]]))

    def step_fn(current_model, current_batch):
        features, target = current_batch
        return (current_model(features) - target).square().mean()

    trainer = Trainer(
        TrainerDependencies(
            model=model,
            optimizer=optimizer,
            train_loader=[batch],
            step_fn=step_fn,
        )
    )
    before = model.weight.detach().clone()

    result = trainer.fit(max_epochs=1, max_train_steps=1)

    assert not torch.equal(before, model.weight)
    assert result.global_step == 1
    assert result.train_loss > 0
