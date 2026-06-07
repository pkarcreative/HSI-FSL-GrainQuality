# evaluate_novel.py
# Evaluates a model trained on a subset of classes (partial class training)
# on the excluded/novel classes it has never seen during training.
# Implements the two evaluation strategies from the paper (Section VI-B).
#
# Strategy 1: Support set contains ONLY the excluded (novel) classes.
#   Simulates a controlled scenario where only the novel classes are present.
#   Expected to give high accuracy as the classification task is simpler.
#
# Strategy 2: Support set contains ALL classes (trained + excluded).
#   Simulates a realistic supply chain scenario where the classifier encounters
#   both known and unknown grain types simultaneously.
#   Performance on novel classes drops due to the larger class space.
#
# Paper result (Rye_Midsummer and Wheat_H5 excluded from training):
#   Strategy 1 accuracy on excluded classes: 98.33%
#   Strategy 2 accuracy on excluded classes: 83.89%
#       Rye_Midsummer: 91.11%   Wheat_H5: 76.67%
#
# Example:
#   python evaluate_novel.py \
#       --test_dir         /path/to/fsl_test \
#       --model_path       /path/to/output_6way/best_model.pth \
#       --classes_file     /path/to/output_6way/classes.txt \
#       --excluded_classes Rye_Midsummer Wheat_H5 \
#       --strategy         both \
#       --k_shot 5 --query_size 10 \
#       --channels 204 --se_attention --reduction_ratio 8

import argparse
import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import HSIDataLoader
from models import ResNet18Backbone, ResNet18BackboneWithSE, PrototypicalNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True,
                        help='Test directory containing ALL classes (trained + excluded)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to best_model.pth from a partial-class training run')
    parser.add_argument('--classes_file', type=str, required=True,
                        help='Path to classes.txt from the partial-class training run')
    parser.add_argument('--excluded_classes', nargs='+', required=True,
                        help='Class names excluded during training (e.g. Rye_Midsummer Wheat_H5)')
    parser.add_argument('--strategy', type=str, default='both',
                        choices=['1', '2', 'both'],
                        help='Evaluation strategy: 1, 2, or both')
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--query_size', type=int, default=10)
    parser.add_argument('--channels', type=int, default=204)
    parser.add_argument('--binning_factor', type=int, default=1)
    parser.add_argument('--se_attention', action='store_true')
    parser.add_argument('--reduction_ratio', type=int, default=8)
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


def compute_protos(model, s_imgs, s_lbls, device):
    s_imgs = s_imgs.to(device)
    s_lbls = s_lbls.to(device)
    feats, _ = model.backbone(s_imgs)
    n_way = len(torch.unique(s_lbls))
    return torch.stack([feats[s_lbls == k].mean(0) for k in range(n_way)])


def print_report(preds, labels, class_names, title):
    acc = (preds == labels).mean() * 100
    print(f'\n--- {title} ---')
    print(f'Accuracy on novel classes: {acc:.2f}%')
    print(classification_report(labels, preds, target_names=class_names, digits=4))
    cm = confusion_matrix(labels, preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    print('Confusion matrix (%):\n', np.round(cm_pct, 2))


def run_strategy1(model, test_dir, excluded_classes, k_shot, query_size, binning_factor, device):
    # Only include excluded classes in the loader.
    all_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    skip = [c for c in all_dirs if c not in excluded_classes]

    loader = HSIDataLoader(
        test_dir, k_shot, query_size,
        exclude_classes=skip,
        binning_factor=binning_factor
    )
    all_preds, all_labels = [], []
    with torch.no_grad():
        for s_imgs, s_lbls, q_imgs, q_lbls in loader:
            protos = compute_protos(model, s_imgs, s_lbls, device)
            q_feats, _ = model.backbone(q_imgs.to(device))
            preds = torch.argmax(-torch.cdist(q_feats, protos), dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(q_lbls.tolist())

    print_report(np.array(all_preds), np.array(all_labels),
                 loader.classes, 'Strategy 1: support = excluded classes only')


def run_strategy2(model, test_dir, excluded_classes, k_shot, query_size, binning_factor, device):
    # All classes in support; filter queries to excluded classes only.
    loader = HSIDataLoader(test_dir, k_shot, query_size, binning_factor=binning_factor)
    excluded_indices = [loader.label_to_idx[c] for c in excluded_classes
                        if c in loader.label_to_idx]
    excl_class_names = [c for c in loader.classes if c in excluded_classes]

    all_preds, all_labels = [], []
    with torch.no_grad():
        for s_imgs, s_lbls, q_imgs, q_lbls in loader:
            protos = compute_protos(model, s_imgs, s_lbls, device)

            mask = torch.zeros(len(q_lbls), dtype=torch.bool)
            for idx in excluded_indices:
                mask |= (q_lbls == idx)
            if mask.sum() == 0:
                continue

            q_feats, _ = model.backbone(q_imgs[mask].to(device))
            preds = torch.argmax(-torch.cdist(q_feats, protos), dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(q_lbls[mask].tolist())

    # remap label indices to 0-based within excluded classes for the report
    idx_map = {orig: new for new, orig in enumerate(excluded_indices)}
    remapped_preds = np.array([idx_map.get(p, p) for p in all_preds])
    remapped_labels = np.array([idx_map[l] for l in all_labels])

    print_report(remapped_preds, remapped_labels,
                 excl_class_names, 'Strategy 2: support = all classes')


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    in_channels = args.channels // args.binning_factor
    model = load_model(args, in_channels, device)

    if args.strategy in ('1', 'both'):
        run_strategy1(model, args.test_dir, args.excluded_classes,
                      args.k_shot, args.query_size, args.binning_factor, device)

    if args.strategy in ('2', 'both'):
        run_strategy2(model, args.test_dir, args.excluded_classes,
                      args.k_shot, args.query_size, args.binning_factor, device)


if __name__ == '__main__':
    main()
