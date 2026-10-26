"""The five symmetric perturbation pairs used during SSA warm-up."""
import math
import torch
import torch.nn.functional as F

PERTURBATIONS = {'brightness': .06, 'contrast': .08, 'saturation': .08,
                 'rotate': 2., 'scale': .04}


def perturb(images, kind, magnitude, sign):
    """Transform RGB images in [0, 1], with shape B x 3 x H x W."""
    factor = 1. + sign * magnitude
    if kind == 'brightness':
        value = images * factor
    elif kind == 'contrast':
        center = images.mean(dim=(1, 2, 3), keepdim=True)
        value = (images-center)*factor+center
    elif kind == 'saturation':
        gray = (.2989*images[:, :1]+.5870*images[:, 1:2]+.1140*images[:, 2:3])
        value = (images-gray)*factor+gray
    elif kind in {'rotate', 'scale'}:
        angle = sign*magnitude*math.pi/180 if kind == 'rotate' else 0.
        scale = factor if kind == 'scale' else 1.
        c, s = math.cos(angle)*scale, math.sin(angle)*scale
        theta = images.new_tensor([[c, -s, 0.], [s, c, 0.]])[None].repeat(len(images), 1, 1)
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        value = F.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    else:
        raise ValueError(f'Unknown perturbation: {kind}')
    return value.clamp(0, 1)


@torch.no_grad()
def response_pairs(encoder, image):
    """Return B x 5 x D normalized-feature differences, (h+ - h-)/2."""
    pixels = (image*encoder.std+encoder.mean).clamp(0, 1)
    views = torch.cat([(perturb(pixels, kind, amount, sign)-encoder.mean)/encoder.std
                       for kind, amount in PERTURBATIONS.items() for sign in (1., -1.)])
    h, _ = encoder.encode(views)
    h = F.normalize(h, dim=-1).reshape(5, 2, len(image), -1)
    return ((h[:, 0]-h[:, 1])/2).permute(1, 0, 2)
