#!/usr/bin/env python3
"""One-crop PyTorch smoke fit for the real preprocessing output; not a research model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from training_data import ROOT, load_sample


class TinyFusionCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv3d(channels, 12, 3, padding=1), nn.GELU(), nn.Conv3d(12, 2, 3, padding=1))

    def forward(self, x):
        return self.net(x)


def crop(sample, size=(8, 32, 32), target_source='query'):
    target_key, mask_key = (('query_target', 'query_mask') if target_source == 'query'
                            else ('target', 'target_mask'))
    d, h, w = sample[target_key].shape[1:]
    cd, ch, cw = (min(a, b) for a, b in zip(size, (d, h, w)))
    best = None
    for z in range(0, d - cd + 1, max(1, cd // 2)):
        for y in range(0, h - ch + 1, max(1, ch // 2)):
            for x in range(0, w - cw + 1, max(1, cw // 2)):
                score = sample[mask_key][:, z:z+cd, y:y+ch, x:x+cw].sum()
                if best is None or score > best[0]:
                    best = (score, z, y, x)
    if best is None or best[0] == 0:
        raise ValueError(f'No valid {target_source} target points in sample')
    _, z, y, x = best
    surface = np.concatenate([sample['surface'], sample['surface_mask']])[:, y:y+ch, x:x+cw]
    position = np.concatenate([sample['sparse_position'], sample['sparse_position_mask']])[:, y:y+ch, x:x+cw]
    surface = np.repeat(np.concatenate([surface, position])[:, None], cd, axis=1)
    sparse = np.concatenate([sample['sparse'], sample['sparse_mask']])[:, z:z+cd, y:y+ch, x:x+cw]
    target = sample[target_key][:, z:z+cd, y:y+ch, x:x+cw]
    mask = sample[mask_key][:, z:z+cd, y:y+ch, x:x+cw]
    return np.concatenate([surface, sparse]), target, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/processed/training_manifest_common.json')
    parser.add_argument('--normalization', type=Path, default=ROOT / 'data/processed/normalization_common.json')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--target-source', choices=['query', 'cora'], default='query')
    parser.add_argument('--output', type=Path, default=ROOT / 'research/audits/smoke_train.json')
    args = parser.parse_args()
    torch.manual_seed(0)
    manifest = json.loads(args.manifest.read_text())
    statistics = json.loads(args.normalization.read_text())['statistics']
    sample = load_sample(manifest['months'][0], statistics)
    inputs, target, mask = [torch.from_numpy(a).float().unsqueeze(0)
                            for a in crop(sample, target_source=args.target_source)]
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    inputs, target, mask = inputs.to(device), target.to(device), mask.to(device)
    model = TinyFusionCNN(inputs.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(args.steps):
        prediction = model(inputs)
        loss = ((prediction - target).square() * mask).sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not np.isfinite(losses).all() or losses[-1] >= losses[0]:
        raise RuntimeError('Smoke fit did not reduce finite masked loss')
    report = {'status': 'passed', 'purpose': 'pipeline smoke test, not model evidence', 'device': device,
              'target_source': args.target_source,
              'steps': args.steps, 'input_shape': list(inputs.shape), 'target_shape': list(target.shape),
              'initial_loss': losses[0], 'final_loss': losses[-1]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
