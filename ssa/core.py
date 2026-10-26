"""Stable Semantic Association (SSA), with bounded streaming state.

The caller supplies one frozen visual representation and Frozen logits per
image. During the first 128 images it also supplies the five symmetric image
perturbation differences. After that no augmented forwards are required.
This adapter never receives medical labels or future images.
"""
import torch
from .geometry import metric_from_second_moment, norm


class InvariantCorrelationStream:
    def __init__(self, dimensions, classes, scale, device='cuda', dtype=torch.float64):
        self.dimensions=dimensions;self.classes=classes;self.scale=float(scale)
        self.n=0;self.warmup=128;self.transform=None
        self.sx=torch.zeros(dimensions,device=device,dtype=dtype)
        self.sp=torch.zeros(classes,device=device,dtype=dtype)
        self.spx=torch.zeros(classes,dimensions,device=device,dtype=dtype)
        self.noise_sum=torch.zeros(dimensions,dimensions,device=device,dtype=dtype)
        self.metric_audit=None

    @torch.no_grad()
    def observe(self, features, frozen_logits, perturbations=None, chunk=64):
        """Adapt on each arrived image and immediately emit its class scores.

        features: B x D, frozen_logits: B x C.
        perturbations: B x K x D, required only for rows before warm-up ends.
        Positive score scaling is kept identical to the fixed experiment;
        these outputs do not claim calibrated clinical probabilities.
        """
        if chunk < 1:
            raise ValueError('chunk must be positive.')
        if features.ndim != 2 or len(features) == 0:
            raise ValueError('Observe at least one feature vector.')
        if not torch.isfinite(features).all() or not torch.isfinite(frozen_logits).all():
            raise ValueError('Features and frozen logits must be finite.')
        if perturbations is not None and not torch.isfinite(perturbations).all():
            raise ValueError('Perturbations must be finite.')
        x=norm(features.to(self.sx));z=frozen_logits.to(self.sx)
        assert x.ndim==2 and x.shape[1]==self.dimensions
        assert z.shape==(len(x),self.classes)
        if self.n<self.warmup:
            needed=min(self.warmup-self.n,len(x))
            if perturbations is None or len(perturbations)<needed:
                raise ValueError('Perturbations for every warm-up image are required.')
        output=[];start=0
        while start<len(x):
            amount=min(chunk,len(x)-start)
            if self.n<self.warmup:amount=min(amount,self.warmup-self.n)
            xx=x[start:start+amount];zz=z[start:start+amount];p=zz.softmax(-1)
            nn=torch.arange(self.n+1,self.n+amount+1,device=x.device,dtype=x.dtype)
            tx=self.sx[None]+xx.cumsum(0)
            tp=self.sp[None]+p.cumsum(0)
            tpx=self.spx[None]+(p[:,:,None]*xx[:,None,:]).cumsum(0)
            mu=tx/nn[:,None]
            g=tpx/nn[:,None,None]-(tp/nn[:,None])[:,:,None]*mu[:,None,:]
            q=xx-mu
            if self.transform is not None:
                g=g@self.transform;q=q@self.transform
            cosine=(norm(g)*norm(q)[:,None,:]).sum(-1)
            score=.5*self.scale*(nn/(nn+512))[:,None]*cosine
            output.append(torch.where((nn>1)[:,None],score,zz))
            if self.n<self.warmup:
                dd=perturbations[start:start+amount].to(self.sx)
                assert dd.ndim==3 and dd.shape[0]==amount and dd.shape[2]==self.dimensions
                flat=dd.reshape(-1,self.dimensions)
                self.noise_sum+=flat.T@flat/dd.shape[1]
            self.sx,self.sp,self.spx=tx[-1].clone(),tp[-1].clone(),tpx[-1].clone()
            self.n+=amount;start+=amount
            # The 128th output above used the unchanged identity metric. This
            # estimate only affects subsequent arrivals.
            if self.n==self.warmup and self.transform is None:
                self.transform,self.metric_audit=metric_from_second_moment(self.noise_sum/self.warmup,self.warmup)
                self.noise_sum=None
        return torch.cat(output)
