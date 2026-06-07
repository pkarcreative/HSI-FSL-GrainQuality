# train.py
# Training script for the FSL prototypical network on HSI grain data.
#
# After training completes, the script automatically runs one additional pass
# through the training episodes using the best saved model to compute and save
# per-episode prototypes. These are used by compute_ccp.py to build CCPs.
#
# Example (8-way):
#   python train.py \
#       --train_dir /path/to/fsl_train \
#       --save_dir  /path/to/output \
#       --k_shot 5 --query_size 10 --epochs 50 \
#       --channels 204 --se_attention --reduction_ratio 8
#
# Example (6-way, exclude Rye and WH5):
#   python train.py \
#       --train_dir /path/to/fsl_train \
#       --save_dir  /path/to/output_6way \
#       --k_shot 5 --query_size 10 --epochs 50 \
#       --channels 204 --se_attention --reduction_ratio 8 \
#       --exclude_classes Rye_Midsummer Wheat_H5

import argparse
import os
import time
import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm
from easyfsl.utils import sliding_average

from dataset import HSIDataLoader
from models import ResNet18Backbone, ResNet18BackboneWithSE, PrototypicalNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory with class subfolders containing .cdf files (FSL training set)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save model weights and per-episode prototypes')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Support samples per class per episode')
    parser.add_argument('--query_size', type=int, default=10,
                        help='Query samples per class per episode')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--channels', type=int, default=204,
                        help='Raw spectral channels after sensor boundary trimming (204 for Specim FX17)')
    parser.add_argument('--binning_factor', type=int, default=1,
                        help='Spectral binning: 1=no binning(204ch), 2=102ch, 3=68ch')
    parser.add_argument('--se_attention', action='store_true',
                        help='Add SE channel attention before spectral downsampling')
    parser.add_argument('--reduction_ratio', type=int, default=8,
                        help='SE block reduction ratio (paper ablation tested 4, 8, 16; best=8)')
    parser.add_argument('--exclude_classes', nargs='*', default=[],
                        help='Class names to exclude (e.g. --exclude_classes Rye_Midsummer Wheat_H5)')
    parser.add_argument('--max_samples_per_class', type=int, default=None,
                        help='Cap on files used per class per epoch (default: use all available)')
    parser.add_argument('--log_freq', type=int, default=10,
                        help='Loss logging frequency (in episodes)')
    return parser.parse_args()


def build_model(args, in_channels, device):
    if args.se_attention:
        backbone = ResNet18BackboneWithSE(in_channels, args.reduction_ratio)
    else:
        backbone = ResNet18Backbone(in_channels)
    return PrototypicalNetwork(backbone).to(device)


def make_loader(args):
    return HSIDataLoader(
        args.train_dir,
        args.k_shot,
        args.query_size,
        exclude_classes=args.exclude_classes,
        max_samples_per_class=args.max_samples_per_class,
        binning_factor=args.binning_factor
    )


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    proto_dir = os.path.join(args.save_dir, 'episode_prototypes')
    os.makedirs(proto_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    in_channels = args.channels // args.binning_factor
    model = build_model(args, in_channels, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # log the class list once so evaluate scripts can load it
    sample_loader = make_loader(args)
    with open(os.path.join(args.save_dir, 'classes.txt'), 'w') as f:
        for cls in sample_loader.classes:
            f.write(cls + '\n')
    print(f'Classes ({sample_loader.n_way}-way): {sample_loader.classes}')
    print(f'Episodes per epoch: {len(sample_loader)}')

    best_loss = float('inf')
    all_loss = []
    start = time.time()

    for epoch in range(args.epochs):
        loader = make_loader(args)
        model.train()
        epoch_losses = []

        with tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch:03d}') as pbar:
            for ep_idx, (s_imgs, s_lbls, q_imgs, q_lbls) in pbar:
                s_imgs = s_imgs.to(device)
                s_lbls = s_lbls.to(device)
                q_imgs = q_imgs.to(device)
                q_lbls = q_lbls.to(device)

                optimizer.zero_grad()
                scores, _ = model(s_imgs, s_lbls, q_imgs)
                loss = criterion(scores, q_lbls)
                loss.backward()
                optimizer.step()

                all_loss.append(loss.item())
                epoch_losses.append(loss.item())
                if ep_idx % args.log_freq == 0:
                    pbar.set_postfix(loss=sliding_average(all_loss, args.log_freq))

        avg_loss = np.mean(epoch_losses)
        print(f'Epoch {epoch:03d} | avg loss: {avg_loss:.4f}')

        torch.save(model.state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch:03d}.pth'))

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f'  -> best model saved (loss {best_loss:.4f})')

    elapsed = (time.time() - start) / 3600
    print(f'\nTraining done in {elapsed:.2f} h')

    # Compute per-episode prototypes using the best model.
    # These are averaged by compute_ccp.py to produce Collective Class Prototypes.
    print('\nComputing per-episode prototypes for CCP...')
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth'), map_location=device))
    model.eval()

    loader = make_loader(args)
    with torch.no_grad():
        for ep_idx, (s_imgs, s_lbls, _, _) in enumerate(loader):
            s_imgs = s_imgs.to(device)
            s_lbls = s_lbls.to(device)
            features, _ = model.backbone(s_imgs)
            n_way = len(torch.unique(s_lbls))
            prototypes = torch.stack([
                features[s_lbls == k].mean(0)
                for k in range(n_way)
            ])
            torch.save(prototypes.cpu(), os.path.join(proto_dir, f'proto_ep_{ep_idx:04d}.pth'))

    n_eps = len([f for f in os.listdir(proto_dir) if f.endswith('.pth')])
    print(f'Saved {n_eps} episode prototype files to {proto_dir}')
    print('Run compute_ccp.py next to produce the final CCPs.')


if __name__ == '__main__':
    main()
