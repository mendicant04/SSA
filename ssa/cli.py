"""Raw-image and cached-feature evaluation of SSA."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .core import InvariantCorrelationStream
from .evaluation import save_run, summarize
CLASSES = json.loads(Path(__file__).with_name('medmnist_classes.json').read_text())

def seeds_arg(value):
    values = [int(x) for x in value.split(',')]
    if not values or len(set(values)) != len(values) or any((v < 0 for v in values)):
        raise argparse.ArgumentTypeError('Use distinct nonnegative integer seeds.')
    return values

def check_output(args, task, seed):
    stem = Path(args.output) / f'{task}_seed{seed}'
    if not args.overwrite and any((stem.with_suffix(s).exists() for s in ('.json', '.npz'))):
        raise FileExistsError(f'{stem} already exists; choose another output folder or pass --overwrite.')

def array(data, *names):
    for key in names:
        if key in data:
            return data[key].copy()
    raise ValueError('Missing cache array; expected one of: ' + ', '.join(names))

def cached(args):
    with np.load(args.input, allow_pickle=False) as data:
        h = array(data, 'h', 'preprojection')
        v = array(data, 'v', 'images')
        text = array(data, 'text')
        ids = data['indices'].copy() if 'indices' in data else np.arange(len(h))
        scale = float(data['scale']) if 'scale' in data else None
        delta = data['delta'].copy() if 'delta' in data else None
        positions = data['probe_positions'].copy() if 'probe_positions' in data else None
    if h.ndim != 2 or v.ndim != 2 or text.ndim != 2 or (len(h) != len(v)) or (v.shape[1] != text.shape[1]):
        raise ValueError('Cache dimensions must be h: N x D, v: N x E, text: C x E.')
    if len(text) < 2 or len(ids) != len(h) or len(np.unique(ids)) != len(ids):
        raise ValueError('Use at least two classes and unique sample IDs.')
    if not all((np.isfinite(x).all() for x in [h, v, text])):
        raise ValueError('Cache features must be finite.')
    if scale is None or not np.isfinite(scale) or scale <= 0:
        raise ValueError('SSA requires the positive native logit scale in the cache.')
    ht = torch.as_tensor(h, device=args.device, dtype=torch.float64)
    vt = F.normalize(torch.as_tensor(v, device=args.device, dtype=torch.float64), dim=-1)
    tt = F.normalize(torch.as_tensor(text, device=args.device, dtype=torch.float64), dim=-1)
    for seed in args.seeds:
        check_output(args, args.task, seed)
        order = np.random.default_rng(seed).permutation(len(h))
        if args.limit:
            order = order[:args.limit]
        if delta is None or delta.ndim != 3 or delta.shape[1] == 0 or (delta.shape[2] != h.shape[1]):
            raise ValueError('SSA requires delta with shape M x K x D.')
        if positions is None:
            if len(delta) != len(h):
                raise ValueError('Provide probe_positions when delta covers only part of the dataset.')
            positions = np.arange(len(h))
        if len(positions) != len(delta) or len(np.unique(positions)) != len(positions):
            raise ValueError('Probe positions must be unique and aligned with delta.')
        lookup = {int(p): i for i, p in enumerate(positions)}
        required = order[:min(128, len(order))]
        missing = [int(p) for p in required if int(p) not in lookup]
        if missing:
            raise ValueError(f'Warm-up probes are missing for {len(missing)} stream positions.')
        dd = torch.as_tensor(delta[[lookup[int(p)] for p in required]], device=args.device, dtype=torch.float64)
        z = float(scale) * vt[order] @ tt.T
        adapter = InvariantCorrelationStream(h.shape[1], len(text), scale, device=args.device)
        scores = adapter.observe(ht[order], z, dd, chunk=args.chunk)
        pred = scores.softmax(-1).argmax(-1).cpu().numpy()
        frozen = z.softmax(-1).argmax(-1).cpu().numpy()
        with np.load(args.input, allow_pickle=False) as data:
            y = array(data, 'y', 'targets') if 'y' in data or 'targets' in data else None
        if y is not None and len(y.reshape(-1)) != len(h):
            raise ValueError('Cache targets must align with feature rows.')
        targets = y.reshape(-1)[order] if y is not None else None
        save_run(args.output, args.task, seed, ids[order], pred, frozen, targets, variant='SSA', mode='cached_features', chunk=args.chunk)

def run_images(args):
    from .backbone import Encoder
    tasks = list(CLASSES) if args.tasks == 'all' else args.tasks.split(',')
    if any((t not in CLASSES for t in tasks)):
        raise ValueError('Unknown MedMNIST task; use the cache API for other datasets.')
    for task in tasks:
        if not (Path(args.data_dir) / f'{task}_224.npz').is_file():
            raise FileNotFoundError(f'Missing dataset: {task}_224.npz under {args.data_dir}')
        for seed in args.seeds:
            check_output(args, task, seed)
    encoder = Encoder(args.model, args.checkpoint_dir, args.device, args.pretrained)
    for task in tasks:
        path = Path(args.data_dir) / f'{task}_224.npz'
        with np.load(path, allow_pickle=False) as data:
            images = data['test_images'].copy()
        text = encoder.text(CLASSES[task], args.template or ['this is a photo of {}'])
        for seed in args.seeds:
            order = np.random.default_rng(seed).permutation(len(images))
            if args.limit:
                order = order[:args.limit]
            adapter = None
            predictions, frozen = ([], [])
            for position, idx in enumerate(order):
                a = images[idx]
                if a.ndim == 3 and a.shape[-1] == 1:
                    a = a[..., 0]
                x = encoder.preprocess(Image.fromarray(a).convert('RGB'))[None].to(args.device)
                h, v = encoder.encode(x)
                if adapter is None:
                    adapter = InvariantCorrelationStream(h.shape[1], len(text), encoder.scale, device=args.device)
                from .probes import response_pairs
                z = encoder.scale * F.normalize(v.double(), dim=-1) @ F.normalize(text.double(), dim=-1).T
                dd = response_pairs(encoder, x) if adapter.n < 128 else None
                score = adapter.observe(h, z, dd, chunk=1)
                predictions.extend(score.softmax(-1).argmax(-1).cpu().tolist())
                frozen.extend(z.softmax(-1).argmax(-1).cpu().tolist())
                if (position + 1) % 512 == 0:
                    print(f'{task} seed={seed}: {position + 1}/{len(order)}', flush=True)
            with np.load(path, allow_pickle=False) as data:
                targets = data['test_labels'].reshape(-1)[order]
            save_run(args.output, task, seed, order, np.asarray(predictions), np.asarray(frozen), targets, variant='SSA', mode='raw_images', encoder_precision='FP32', statistics_precision='FP64', model=args.model if not args.checkpoint_dir else 'local OpenCLIP checkpoint', templates=args.template or ['this is a photo of {}'])

def main():
    parser = argparse.ArgumentParser(description='SSA online test-time adaptation')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['run', 'cache']:
        p = sub.add_parser(name, help='Evaluate raw MedMNIST images' if name == 'run' else 'Evaluate cached frozen features')
        p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
        p.add_argument('--seeds', type=seeds_arg, default=[0, 1, 2])
        p.add_argument('--output', default='outputs/ssa')
        p.add_argument('--limit', type=int, default=0, help='0 evaluates the complete stream; positive values select a prefix')
        p.add_argument('--threads', type=int, default=2)
        p.add_argument('--overwrite', action='store_true')
        if name == 'cache':
            p.add_argument('--input', required=True)
            p.add_argument('--task', required=True, help='Stable dataset name, also used for deterministic LERN tie breaking')
            p.add_argument('--chunk', type=int, default=64)
        else:
            from .backbone import BIOMEDCLIP
            p.add_argument('--data-dir', default='data/medmnist')
            p.add_argument('--tasks', default='all')
            p.add_argument('--model', default=BIOMEDCLIP)
            p.add_argument('--pretrained', default=None)
            p.add_argument('--checkpoint-dir', default=None, help='Local OpenCLIP checkpoint and tokenizer directory')
            p.add_argument('--template', action='append', help='Text template containing {}; repeat for an ensemble')
    p = sub.add_parser('summarize', help='Compute equal-weight per-dataset and macro metrics')
    p.add_argument('--input', required=True, help='Directory containing scored run JSON files')
    args = parser.parse_args()
    if args.command == 'summarize':
        summarize(args.input)
        return
    if args.limit < 0 or args.threads < 1 or getattr(args, 'chunk', 1) < 1:
        parser.error('limit must be nonnegative; threads and chunk must be positive')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == 'cache':
        cached(args)
    else:
        run_images(args)
