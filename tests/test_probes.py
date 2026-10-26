import unittest
import torch
from ssa.probes import PERTURBATIONS, perturb

class ProbeTests(unittest.TestCase):
    def test_each_image_is_perturbed_independently(self):
        x=torch.rand(3,3,16,16)
        for kind,amount in PERTURBATIONS.items():
            for sign in [-1,1]:
                batch=perturb(x,kind,amount,sign)
                separate=torch.cat([perturb(row[None],kind,amount,sign) for row in x])
                torch.testing.assert_close(batch,separate)
                self.assertTrue(torch.isfinite(batch).all())
                self.assertGreaterEqual(float(batch.min()),0.)
                self.assertLessEqual(float(batch.max()),1.)

if __name__=='__main__':unittest.main()
