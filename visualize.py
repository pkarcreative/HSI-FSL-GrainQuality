# visualize.py
# Produces the three visualisations from the paper:
#   1. t-SNE scatter plot of episode prototypes and CCPs (Fig. 8)
#   2. SE attention weight heatmap across spectral channels (Fig. 3)
#   3. Confusion matrix as a percentage heatmap (Figs. 5/6)
#
# Each plot is saved as a .png file in --output_dir.
#
# Example:
#   python visualize.py \
#       --test_dir     /path/to/fsl_test \
#       --model_path   /path/to/output/best_model.pth \
#       --ccp_path     /path/to/output/ccp.pth \
#       --proto_dir    /path/to/output/episode_prototypes \
#       --classes_file /path/to/output/classes.txt \
#       --output_dir   /path/to/figures \
#       --channels 204 --se_attention --reduction_ratio 8 \
#       --k_shot 5 --query_size 10

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from dataset import HSIDataLoader
from models import ResNet18Backbone, ResNet18BackboneWithSE, PrototypicalNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--classes_file', type=str, required=True)
    parser.add_argument('--ccp_path', type=str, default=None,
                        help='ccp.pth for t-SNE and inference. Required for t-SNE and confusion matrix.')
    parser.add_argument('--proto_dir', type=str, default=None,
                        help='episode_prototypes/ dir for t-SNE scatter. Required for t-SNE.')
    parser.add_argument('--output_dir', type=str, default='figures')
    parser.add_argument('--channels', type=int, default=204)
    parser.add_argument('--binning_factor', type=int, default=1)
    parser.add_argument('--se_attention', action='store_true')
    parser.add_argument('--reduction_ratio', type=int, default=8)
    parser.add_argument('--k_shot', type=int, default=5)
    parser.add_argument('--query_size', type=int, default=10)
    parser.add_argument('--tsne_perplexity', type=int, default=30)
    parser.add_argument('--exclude_classes', nargs='*', default=[])
    return parser.parse_args()


def load_model(args, device):
    in_channels = args.channels // args.binning_factor
    if args.se_attention:
        backbone = ResNet18BackboneWithSE(in_channels, args.reduction_ratio)
    else:
        backbone = ResNet18Backbone(in_channels)
    model = PrototypicalNetwork(backbone).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    return model


def plot_tsne(proto_dir, ccp_path, classes, output_dir, perplexity):
    # Load all episode prototypes and CCPs, reduce to 2D with t-SNE, plot.
    if proto_dir is None or ccp_path is None:
        print('Skipping t-SNE: --proto_dir and --ccp_path both required')
        return

    files = sorted([f for f in os.listdir(proto_dir) if f.endswith('.pth')])
    ep_protos = [torch.load(os.path.join(proto_dir, f), map_location='cpu').numpy()
                 for f in files]
    ccp = torch.load(ccp_path, map_location='cpu').numpy()

    n_way = ccp.shape[0]
    n_eps = len(ep_protos)

    # stack all points: episode prototypes + CCPs
    all_points = np.vstack(ep_protos + [ccp])  # (n_eps*n_way + n_way, d)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(all_points)

    ep_reduced = reduced[:n_eps * n_way].reshape(n_eps, n_way, 2)
    ccp_reduced = reduced[n_eps * n_way:]

    colors = plt.cm.tab10(np.linspace(0, 1, n_way))
    fig, ax = plt.subplots(figsize=(9, 7))

    for k in range(n_way):
        ax.scatter(ep_reduced[:, k, 0], ep_reduced[:, k, 1],
                   color=colors[k], s=20, alpha=0.5, label=None)
        ax.scatter(ccp_reduced[k, 0], ccp_reduced[k, 1],
                   color=colors[k], s=150, marker='*',
                   edgecolors='black', linewidths=0.5, label=classes[k])

    ax.legend(fontsize=8, loc='best')
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title('Prototypes (circles) and CCPs (stars) in 2D t-SNE space')
    plt.tight_layout()

    out = os.path.join(output_dir, 'tsne_prototypes.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


def plot_attention_heatmap(model, loader, classes, output_dir, device):
    # Run one episode through the model, collect SE attention weights,
    # average per class and plot as a heatmap (channels x classes).
    model.eval()
    class_attn = {cls: [] for cls in classes}

    with torch.no_grad():
        for s_imgs, s_lbls, _, _ in loader:
            s_imgs = s_imgs.to(device)
            _, attn = model.backbone(s_imgs)
            if attn is None:
                print('Skipping attention heatmap: model has no SE attention (use --se_attention)')
                return
            # attn shape: (batch, channels, 1, 1)
            attn = attn.squeeze(-1).squeeze(-1).cpu().numpy()
            for i, lbl in enumerate(s_lbls.tolist()):
                class_attn[classes[lbl]].append(attn[i])
            break  # one episode is enough for visualisation

    avg_attn = np.array([np.mean(class_attn[cls], axis=0)
                         for cls in classes
                         if class_attn[cls]])

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(avg_attn, ax=ax, cmap='viridis',
                xticklabels=range(0, avg_attn.shape[1], max(1, avg_attn.shape[1] // 20)),
                yticklabels=classes)
    ax.set_xlabel('Channel index')
    ax.set_ylabel('Class')
    ax.set_title('SE attention weight distribution across spectral channels')
    plt.tight_layout()

    out = os.path.join(output_dir, 'attention_heatmap.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


def plot_confusion_matrix(model, loader, ccp, classes, output_dir, device):
    if ccp is None:
        print('Skipping confusion matrix: --ccp_path required')
        return

    all_preds, all_labels = [], []
    ccp_t = ccp.to(device)
    model.eval()
    with torch.no_grad():
        for _, _, q_imgs, q_lbls in loader:
            feats, _ = model.backbone(q_imgs.to(device))
            preds = torch.argmax(-torch.cdist(feats, ccp_t), dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(q_lbls.tolist())

    cm = confusion_matrix(all_labels, all_preds)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_pct, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion matrix (%)')
    plt.tight_layout()

    out = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.classes_file) as f:
        classes = [l.strip() for l in f if l.strip()]

    model = load_model(args, device)

    loader = HSIDataLoader(
        args.test_dir, args.k_shot, args.query_size,
        exclude_classes=args.exclude_classes,
        binning_factor=args.binning_factor
    )

    ccp = torch.load(args.ccp_path, map_location='cpu') if args.ccp_path else None

    plot_tsne(args.proto_dir, args.ccp_path, classes, args.output_dir, args.tsne_perplexity)
    plot_attention_heatmap(model, loader, classes, args.output_dir, device)
    plot_confusion_matrix(model, loader, ccp, classes, args.output_dir, device)


if __name__ == '__main__':
    main()
