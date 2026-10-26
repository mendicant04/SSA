"""Regularized response geometry for SSA."""
import torch
import torch.nn.functional as F

def norm(x):
    return F.normalize(x, dim=-1, eps=1e-12)

def metric_from_second_moment(cov, n):
    """Standard OAS-form isotropic regularization, n = independent images.

Perturbations from the same image are not counted as independent samples.
OAS is a regularizer here; we make no claim that these tangent probes satisfy
its Gaussian optimality assumptions.
"""
    d = len(cov); mu = cov.trace() / d
    if float(mu) < 1e-20:
        return torch.eye(d,device=cov.device,dtype=cov.dtype), dict(shrinkage=1.,condition_number=1.)
    alpha = cov.square().mean()
    numerator = alpha + mu.square()
    denominator = (n+1)*(alpha - mu.square()/d)
    shrink = torch.clamp(numerator/denominator.clamp_min(1e-30), 0., 1.)
    regular = (1-shrink)*cov + shrink*mu*torch.eye(d,device=cov.device,dtype=cov.dtype)
    eig, vec = torch.linalg.eigh(regular / mu)
    eig = eig.clamp_min(1e-12)
    transform = vec * eig.rsqrt()[None,:]
    return transform, dict(shrinkage=float(shrink), condition_number=float(eig[-1]/eig[0]),
                           raw_mean_energy=float(mu), independent_images=n)
