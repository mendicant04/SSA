"""Small CPU example; no dataset or checkpoint is needed."""
import numpy as np
import torch
from ssa import SSA

torch.set_num_threads(2)
rng=np.random.default_rng(0)
h=rng.normal(size=(160,16))
v=rng.normal(size=(160,8))
text=rng.normal(size=(4,8))

adapter=SSA(16,4,scale=20.,device='cpu')
logits=torch.tensor(rng.normal(size=(160,4)),dtype=torch.float64)
delta=torch.tensor(rng.normal(size=(128,5,16)),dtype=torch.float64)
predictions=[]
for i in range(len(h)):
    probes=delta[i:i+1] if i<128 else None
    scores=adapter.observe(torch.tensor(h[i:i+1]),logits[i:i+1],probes,chunk=1)
    predictions.append(int(scores.argmax(-1)[0]))
print('Processed',adapter.n,'images. First predictions:',predictions[:10])
