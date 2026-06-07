# compute_ccp.py
# Computes Collective Class Prototypes (CCP) by averaging per-episode prototypes
# saved by train.py across all training episodes.
#
# CCPs are the paper's core contribution for robust inference:
# - individual support sets can contain outliers that bias prototypes
# - averaging over all training episodes produces stable class representations
# - CCPs eliminate the need to provide support images at inference time
#
# Run after train.py has finished:
#   python compute_ccp.py \
#       --save_dir /path/to/output \
#       --output   /path/to/output/ccp.pth

import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory used as --save_dir in train.py '
                             '(must contain episode_prototypes/ subfolder)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to write ccp.pth (default: <save_dir>/ccp.pth)')
    return parser.parse_args()


def main():
    args = parse_args()
    proto_dir = os.path.join(args.save_dir, 'episode_prototypes')

    if not os.path.isdir(proto_dir):
        raise FileNotFoundError(f'episode_prototypes/ not found under {args.save_dir}. '
                                'Run train.py first.')

    files = sorted([f for f in os.listdir(proto_dir) if f.endswith('.pth')])
    if not files:
        raise FileNotFoundError(f'No .pth files found in {proto_dir}')

    all_protos = []
    for fname in files:
        p = torch.load(os.path.join(proto_dir, fname), map_location='cpu')
        all_protos.append(p)

    # stacked shape: (num_episodes, n_way, feature_dim)
    stacked = torch.stack(all_protos, dim=0)
    ccp = stacked.mean(dim=0)   # (n_way, feature_dim)

    out_path = args.output or os.path.join(args.save_dir, 'ccp.pth')
    torch.save(ccp, out_path)
    print(f'CCP saved: {out_path}')
    print(f'  Episodes averaged : {len(files)}')
    print(f'  CCP shape         : {list(ccp.shape)}  [n_way x feature_dim]')


if __name__ == '__main__':
    main()
