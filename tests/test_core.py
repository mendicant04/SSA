import unittest
import torch
import torch.nn.functional as F
from ssa import SSA

class SSATests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        g=torch.Generator().manual_seed(33)
        self.h=F.normalize(torch.randn(173,13,generator=g,dtype=torch.float64),dim=-1)
        self.z=torch.randn(173,4,generator=g,dtype=torch.float64)
        self.delta=torch.randn(128,5,13,generator=g,dtype=torch.float64)

    def run_stream(self,n=173,chunk=1,z=None):
        a=SSA(13,4,20.,device='cpu')
        return a,a.observe(self.h[:n],(self.z if z is None else z)[:n],self.delta[:min(n,128)],chunk=chunk)

    def test_chunking_and_prefix(self):
        _,reference=self.run_stream()
        for chunk in [7,64,150]:
            _,result=self.run_stream(chunk=chunk)
            torch.testing.assert_close(result,reference,rtol=1e-9,atol=1e-9)
        _,prefix=self.run_stream(n=149,chunk=31)
        torch.testing.assert_close(prefix,reference[:149],rtol=1e-9,atol=1e-9)

    def test_explicit_prefix_objective(self):
        adapter,scores=self.run_stream(chunk=19)
        for i in [1,36,127,128,172]:
            x=self.h[:i+1];p=self.z[:i+1].softmax(-1);mean=x.mean(0)
            g=p.T@x/(i+1)-p.mean(0)[:,None]*mean
            q=self.h[i]-mean
            if i>=128:g=g@adapter.transform;q=q@adapter.transform
            expected=10.*(i+1)/(i+1+512)*(F.normalize(g,dim=-1)*F.normalize(q,dim=-1)).sum(-1)
            torch.testing.assert_close(scores[i],expected,rtol=1e-8,atol=1e-8)
        torch.testing.assert_close(scores[0],self.z[0])
        self.assertIsNone(adapter.noise_sum)

    def test_class_permutation(self):
        perm=torch.tensor([2,0,3,1])
        _,a=self.run_stream();_,b=self.run_stream(z=self.z[:,perm])
        torch.testing.assert_close(b,a[:,perm],rtol=1e-9,atol=1e-9)

    def test_missing_probes_and_zero_chunk(self):
        a=SSA(13,4,20.,device='cpu')
        with self.assertRaises(ValueError):a.observe(self.h[:1],self.z[:1])
        with self.assertRaises(ValueError):a.observe(self.h[:1],self.z[:1],self.delta[:1],chunk=0)

if __name__=='__main__':unittest.main()
