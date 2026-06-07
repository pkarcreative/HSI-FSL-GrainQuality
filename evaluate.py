# evaluate.py
# Evaluates the trained model on the test set.
#
# Two inference modes:
#   1. CCP mode (--ccp_path): uses pre-computed Collective Class Prototypes.
#      No support images needed at inference. Faster and more robust.
#      Run compute_ccp.py first to produce ccp.pth.
#
#   2. Support-set mode (default, no --ccp_path): each test episode provides
#      its own k_shot support images which are used to compute live prototypes.
#      Accuracy varies across episodes due to support set outliers.
#
# Example (CCP inference):
#   python evaluate.py \
#       --test_dir   /path/to/fsl_test \
#       --model_path /path/to/output/best_model.pth \
#       --ccp_path   /path/to/output/ccp.pth \
#       --classes_file /path/to/output/classes.txt \
#       --k_shot 5 --query_size 10 \
#       --channels 204 --se_attention --reduction_ratio 8
#
# Example (support-set inference):
#   python evaluate.py \
#       --test_dir   /path/to/fsl_test \
#       --model_path /path/to/output/best_model.pth \
#       --classes_file /path/to/output/classes.txt \
#       --k_shot 5 --query_size 10 \
#       --channels 204 --se_attention

import argparse
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import HSIDataLoader
from models import ResNet18Backbone, ResNet18BackboneWithSE, PrototypicalNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Test directory with class subfolders containing .cdf files')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to best_model.pth saved by train.py')
    parser.add_argument('--classes_file', type=str, required=True,
                        help='Path to classes.txt saved by train.py')
    parser.add_argument('--ccp_path', type=str, default=None,
                        help='Path to ccp.pth from compute_ccp.py. '
                             'If omitted, evaluation uses live support sets.')
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--query_size', type=int, default=10)
    parser.add_argument('--channels', type=int, default=204)
    parser.add_argument('--binning_factor', type=int, default=1)
    parser.add_argument('--se_attention', action='store_true')
    parser.add_argument('--reduction_ratio', type=int, default=8)
    parser.add_argument('--exclude_classes', nargs='*', default=[])
    return parser.parse_args()


def load_model(args, in_channels, device):
    if args.se_attention:
        backbone = ResNet18BackboneWithSE(in_channels, args.reduction_ratio)
    else:
        backbone = ResNet18Backbone(in_channels)
    model = PrototypicalNetwork(backbone).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    return model


def eval_ccp(model, loader, ccps, device):
    all_preds, all_labels = [], []
    ccps = ccps.to(device)
    with torch.no_grad():
        for _, _, q_imgs, q_lbls in loader:
            q_imgs = q_imgs.to(device)
            features, _ = model.backbone(q_imgs)
            dists = torch.cdist(features, ccps)
            preds = torch.argmax(-dists, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(q_lbls.tolist())
    return np.array(all_preds), np.array(all_labels)


def eval_support_sets(model, loader, device):
    # For each episode, compute prototypes from the episode's own support set
    # and classify the episode's query images.
    all_preds, all_labels = [], []
    ep_accuracies = []
    with torch.no_grad():
        for s_imgs, s_lbls, q_imgs, q_lbls in loader:
            s_imgs = s_imgs.to(device)
            s_lbls = s_lbls.to(device)
            q_imgs = q_imgs.to(device)

            s_feats, _ = model.backbone(s_imgs)
            n_way = len(torch.unique(s_lbls))
            protos = torch.stack([s_feats[s_lbls == k].mean(0) for k in range(n_way)])

            q_feats, _ = model.backbone(q_imgs)
            dists = torch.cdist(q_feats, protos)
            preds = torch.argmax(-dists, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(q_lbls.tolist())
            ep_acc = (preds.cpu() == q_lbls).float().mean().item() * 100
            ep_accuracies.append(ep_acc)

    print(f'Per-episode accuracy: mean={np.mean(ep_accuracies):.2f}%  '
          f'std={np.std(ep_accuracies):.2f}%  '
          f'min={np.min(ep_accuracies):.2f}%  '
          f'max={np.max(ep_accuracies):.2f}%')
    return np.array(all_preds), np.array(all_labels)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.classes_file) as f:
        classes = [l.strip() for l in f if l.strip()]

    in_channels = args.channels // args.binning_factor
    model = load_model(args, in_channels, device)

    loader = HSIDataLoader(
        args.test_dir,
        args.k_shot,
        args.query_size,
        exclude_classes=args.exclude_classes,
        binning_factor=args.binning_factor
    )

    if args.ccp_path:
        ccps = torch.load(args.ccp_path, map_location='cpu')
        print(f'CCP inference  | CCP shape: {list(ccps.shape)}')
        preds, labels = eval_ccp(model, loader, ccps, device)
    else:
        print('Support-set inference')
        preds, labels = eval_support_sets(model, loader, device)

    acc = (preds == labels).mean() * 100
    print(f'\nOverall accuracy: {acc:.2f}%')
    print(classification_report(labels, preds, target_names=classes, digits=4))

    cm = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    print('Confusion matrix (%):\n', np.round(cm_pct, 2))


if __name__ == '__main__':
    main()
