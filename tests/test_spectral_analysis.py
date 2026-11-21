import numpy as np
import pytest
import torch

from mmrecsys.experiment.spectral_analysis import spectral_metrics


def test_uniform_and_concentrated_power():
    z = torch.ones(2, 8, dtype=torch.complex128)
    m = spectral_metrics(z, z)
    np.testing.assert_allclose(m['image_entropy'], 3)
    np.testing.assert_allclose(m['image_energy_top0.25'], .25)
    np.testing.assert_allclose(m['phase_variance'], 0)
    z[:, 1:] = 0
    m = spectral_metrics(z, z)
    np.testing.assert_allclose(m['image_entropy'], 0)
    np.testing.assert_allclose(m['image_energy_top0.25'], 1)


def test_phase_population_variance_and_scale_invariance():
    a = torch.tensor([[0., .5, -1., 2.]], dtype=torch.float64)
    b = torch.tensor([[1., -.5, 2., -2.]], dtype=torch.float64)
    image, text = torch.exp(1j*a), torch.exp(1j*b)
    m = spectral_metrics(image, text)
    np.testing.assert_allclose(m['phase_variance'], np.var((b-a).numpy(), axis=-1))
    scaled = spectral_metrics(image*7, text*3)
    for key in m:
        np.testing.assert_allclose(m[key], scaled[key], atol=1e-12)


def test_zero_energy_rejected():
    with pytest.raises(ValueError, match='zero-energy'):
        spectral_metrics(torch.zeros(1, 8, dtype=torch.complex64), torch.ones(1, 8, dtype=torch.complex64))
