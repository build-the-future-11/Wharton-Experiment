"""Small finite-difference checks and the actual predictive training path.

These fixtures test derivatives and routing, not prediction performance.
"""
from unittest.mock import patch

import numpy as np
import torch

from wharton_lab.models.m06.config import M06Config
from wharton_lab.models.m06.model import FIJEPAModel
from wharton_lab.models.m06.modules import vicreg_loss
from wharton_lab.models.m07.psd import ConditionalPSDCovHead


def test_vicreg_double_precision_finite_differences():
    # Variances stay away from the ReLU kink at standard deviation one.
    z = torch.tensor([[.1, .3], [-.2, .1], [.4, -.3], [-.1, -.2]],
                     dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(vicreg_loss, (z,), eps=1e-6,
                                    atol=1e-5, rtol=1e-3)


def test_conditional_covariance_input_and_parameter_finite_differences():
    torch.manual_seed(2026101906)
    head = ConditionalPSDCovHead(2, 2).double()
    context = torch.tensor([[.2, -.4], [.7, .1]], dtype=torch.float64,
                           requires_grad=True)
    names, parameters = zip(*head.named_parameters())

    def forward(x, *values):
        return torch.func.functional_call(head, dict(zip(names, values)), (x,))

    assert torch.autograd.gradcheck(forward, (context, *parameters), eps=1e-6,
                                    atol=1e-5, rtol=1e-3)
    assert torch.linalg.eigvalsh(head(context)).min() > 0


def test_actual_m06_predictive_loss_routes_gradients_and_detaches_target():
    rng = np.random.default_rng(2026101906)
    context = rng.normal(size=(10, 3, 2))
    future = rng.normal(size=(10, 1, 2))
    model = FIJEPAModel(M06Config(seq_len=3, input_dim=2, hidden_dim=4,
        latent_dim=3, horizons=(1,), epochs=1, batch_size=16,
        aux_coeff=0, vicreg_coeff=0, device="cpu", random_state=2026101906))
    observed = []
    original_step = model._opt.step

    def audited_step(*args, **kwargs):
        norms = {}
        for name, module in (("context", model.context),
                             ("predictor", model.predictors["1"])):
            grads = [p.grad for p in module.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            norms[name] = sum(float(g.norm()) for g in grads)
            assert norms[name] > 0
        assert all(p.grad is None and not p.requires_grad for p in model.target.parameters())
        observed.append(norms)
        return original_step(*args, **kwargs)

    with patch.object(model._opt, "step", side_effect=audited_step):
        model.fit(context, np.zeros(10), future_windows={1: future}, predictive_coeff=1)
    assert len(observed) == 1
