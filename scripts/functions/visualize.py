import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except Exception:
    try:
        plt.style.use('seaborn-whitegrid')
    except Exception:
        pass

COLOR_PALETTE = ['#2C3E50', '#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6']

LOSS_COMPONENT_COLORS = {
    'info_nce': '#2C3E50',
    'occ': '#E74C3C',
    'covariance': '#3498DB',
    'empathy_ortho': '#27AE60',
    'rl_policy': '#F39C12',
    'cls': '#9B59B6'
}

ROUTING_COLORS = ['#3498DB', '#E74C3C', '#27AE60']
ROUTING_LABELS = ['GNN Expert', 'LLM Expert', 'Vision Expert']

STAGE1_SHADE_COLOR = '#EAEAEA'
STAGE2_SHADE_COLOR = '#C8C8C8'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'lines.linewidth': 2.0,
    'lines.markersize': 5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '-',
})


class TrainingVisualizer:
    def __init__(self, log_dir: str = "logs", experiment_name: str = "EASE-MoE"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.val_accuracies: List[float] = []
        self.val_precisions: List[float] = []
        self.val_recalls: List[float] = []
        self.val_f1_scores: List[float] = []
        self.val_auc_scores: List[float] = []
        self.learning_rates: List[float] = []
        self.epochs: List[int] = []
        self.stages: List[int] = []

        self.routing_weights_history: List[Dict] = []

        self.loss_components: Dict[str, List[float]] = {
            'info_nce': [],
            'occ': [],
            'covariance': [],
            'empathy_ortho': [],
            'rl_policy': [],
            'cls': []
        }

        self.batch_steps: List[int] = []
        self.batch_losses: List[float] = []
        self.batch_accs: List[float] = []
        self.batch_loss_components: Dict[str, List[float]] = {
            'info_nce': [], 'occ': [], 'covariance': [],
            'empathy_ortho': [], 'rl_policy': [], 'cls': []
        }
        self.batch_routing_weights: List[Dict] = []
        self.batch_stages: List[int] = []

    def log_batch(
        self,
        global_step: int,
        batch_loss: float,
        batch_acc: float,
        loss_components: Dict[str, float] = None,
        routing_weights: Dict[str, float] = None,
        stage: int = 1
    ):
        self.batch_steps.append(global_step)
        self.batch_losses.append(batch_loss)
        self.batch_accs.append(batch_acc)
        self.batch_stages.append(stage)

        if loss_components:
            for key, value in loss_components.items():
                if key in self.batch_loss_components:
                    self.batch_loss_components[key].append(value)
        if routing_weights:
            self.batch_routing_weights.append(routing_weights)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float = None,
        train_acc: float = None,
        val_acc: float = None,
        val_precision: float = None,
        val_recall: float = None,
        val_f1: float = None,
        val_auc: float = None,
        lr: float = None,
        stage: int = 1,
        loss_components: Dict[str, float] = None,
        routing_weights: Dict[str, float] = None
    ):
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.stages.append(stage)

        if val_loss is not None:
            self.val_losses.append(val_loss)
        if train_acc is not None:
            self.train_accuracies.append(train_acc)
        if val_acc is not None:
            self.val_accuracies.append(val_acc)
        if val_precision is not None:
            self.val_precisions.append(val_precision)
        if val_recall is not None:
            self.val_recalls.append(val_recall)
        if val_f1 is not None:
            self.val_f1_scores.append(val_f1)
        if val_auc is not None:
            self.val_auc_scores.append(val_auc)
        if lr is not None:
            self.learning_rates.append(lr)

        if loss_components:
            for key, value in loss_components.items():
                if key in self.loss_components:
                    self.loss_components[key].append(value)

        if routing_weights:
            self.routing_weights_history.append(routing_weights)

    def _get_best_f1_info(self):
        if not self.val_f1_scores:
            return None, None, None
        best_idx = int(np.argmax(self.val_f1_scores))
        best_f1 = self.val_f1_scores[best_idx]
        best_epoch = self.epochs[best_idx] if best_idx < len(self.epochs) else best_idx + 1
        return best_epoch, best_f1, best_idx

    def _annotate_best_f1(self, ax, best_epoch, best_f1, y_pos=None):
        if best_epoch is None:
            return
        ax.axvline(x=best_epoch, color='#E74C3C', linestyle='--', linewidth=1.2, alpha=0.6)
        if y_pos is None:
            ylim = ax.get_ylim()
            y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.92
        ax.annotate(
            f'Best F1: {best_f1:.3f}\n(Epoch {best_epoch})',
            xy=(best_epoch, y_pos),
            fontsize=8,
            color='#E74C3C',
            ha='center',
            va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#E74C3C', alpha=0.85, linewidth=0.8)
        )

    def _get_stage_boundary(self):
        if not self.stages:
            return None
        for i in range(1, len(self.stages)):
            if self.stages[i] != self.stages[i - 1]:
                return self.epochs[i - 1]
        return None

    def _get_batch_stage_boundary(self):
        if not self.batch_stages:
            return None
        for i in range(1, len(self.batch_stages)):
            if self.batch_stages[i] != self.batch_stages[i - 1]:
                return self.batch_steps[i - 1]
        return None

    def _add_stage_shading(self, ax):
        boundary = self._get_stage_boundary()
        if boundary is not None and self.epochs:
            x_min = self.epochs[0] - 0.5
            ax.axvspan(x_min, boundary + 0.5, alpha=0.25,
                       color=STAGE1_SHADE_COLOR, zorder=0)
            ax.axvspan(boundary + 0.5, self.epochs[-1] + 0.5, alpha=0.25,
                       color=STAGE2_SHADE_COLOR, zorder=0)
            ax.axvline(x=boundary + 0.5, color='#888888', linestyle='--',
                       linewidth=1.2, alpha=0.5)

            stage_patches = [
                mpatches.Patch(facecolor=STAGE1_SHADE_COLOR, alpha=0.5,
                               label='Stage 1 (Labeled)'),
                mpatches.Patch(facecolor=STAGE2_SHADE_COLOR, alpha=0.5,
                               label='Stage 2 (Semi-supervised)')
            ]
            existing_legend = ax.get_legend()
            if existing_legend:
                existing_handles = existing_legend.legend_handles
                existing_labels = [t.get_text() for t in existing_legend.get_texts()]
                for p in stage_patches:
                    lbl = p.get_label()
                    if lbl not in existing_labels:
                        existing_handles.append(p)
                        existing_labels.append(lbl)
                ax.legend(handles=existing_handles, labels=existing_labels,
                          fontsize=9, loc='best', frameon=True,
                          fancybox=True, framealpha=0.9)
            else:
                ax.legend(handles=stage_patches, fontsize=9, loc='best',
                          frameon=True, fancybox=True, framealpha=0.9)

    def _add_batch_stage_shading(self, ax):
        boundary = self._get_batch_stage_boundary()
        if boundary is not None and self.batch_steps:
            x_min = self.batch_steps[0] - 0.5
            ax.axvspan(x_min, boundary + 0.5, alpha=0.20,
                       color=STAGE1_SHADE_COLOR, zorder=0)
            ax.axvspan(boundary + 0.5, self.batch_steps[-1] + 0.5, alpha=0.20,
                       color=STAGE2_SHADE_COLOR, zorder=0)
            ax.axvline(x=boundary + 0.5, color='#888888', linestyle='--',
                       linewidth=1.2, alpha=0.5)

    def generate_all_plots(self) -> Dict[str, str]:
        results = {}
        results['batch_loss'] = self._plot_batch_loss()
        results['batch_acc'] = self._plot_batch_acc()
        results['batch_loss_components'] = self._plot_batch_loss_components()
        results['batch_routing_weights'] = self._plot_batch_routing_weights()
        results['loss_curves'] = self._plot_loss_curves()
        results['accuracy_curves'] = self._plot_accuracy_curves()
        results['metrics_curves'] = self._plot_metrics_curves()
        results['loss_components'] = self._plot_loss_components()
        results['routing_weights'] = self._plot_routing_weights()
        return {k: v for k, v in results.items() if v is not None}

    def _plot_batch_loss(self):
        if not self.batch_losses:
            return None
        save_path = os.path.join(self.log_dir, f"{self.experiment_name}_batch_loss.png")

        fig, ax = plt.subplots(figsize=(16, 6))
        self._add_batch_stage_shading(ax)

        window = min(10, max(1, len(self.batch_losses) // 20))
        if len(self.batch_losses) > window:
            smooth = np.convolve(self.batch_losses, np.ones(window) / window, mode='valid')
            smooth_steps = self.batch_steps[window - 1:]
            ax.plot(self.batch_steps, self.batch_losses, '-',
                    color=COLOR_PALETTE[0], alpha=0.20, linewidth=0.6,
                    label='Batch Loss')
            ax.plot(smooth_steps, smooth, '-', color=COLOR_PALETTE[0],
                    linewidth=2.0, label=f'Moving Avg (w={window})')
        else:
            ax.plot(self.batch_steps, self.batch_losses, '-',
                    color=COLOR_PALETTE[0], linewidth=2.0, label='Batch Loss')

        epoch_starts = []
        seen_epochs = set()
        for step in self.batch_steps:
            epoch_num = step // 1000 + 1
            if epoch_num not in seen_epochs:
                seen_epochs.add(epoch_num)
                epoch_starts.append(step)
        for es in epoch_starts:
            ax.axvline(x=es, color='#AAAAAA', linestyle=':', linewidth=0.8, alpha=0.4)

        ax.set_xlabel('Training Step (Batch)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{self.experiment_name} \u2014 Batch-Level Training Loss', fontsize=14)
        ax.legend(fontsize=10, loc='upper right', frameon=True,
                  fancybox=True, framealpha=0.9)
        plt.tight_layout(pad=1.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path

    def _plot_batch_acc(self):
        if not self.batch_accs:
            return None
        save_path = os.path.join(self.log_dir, f"{self.experiment_name}_batch_acc.png")

        fig, ax = plt.subplots(figsize=(16, 6))
        self._add_batch_stage_shading(ax)

        window = min(10, max(1, len(self.batch_accs) // 20))
        if len(self.batch_accs) > window:
            smooth = np.convolve(self.batch_accs, np.ones(window) / window, mode='valid')
            smooth_steps = self.batch_steps[window - 1:]
            ax.plot(self.batch_steps, self.batch_accs, '-',
                    color=COLOR_PALETTE[3], alpha=0.20, linewidth=0.6,
                    label='Batch Accuracy')
            ax.plot(smooth_steps, smooth, '-', color=COLOR_PALETTE[3],
                    linewidth=2.0, label=f'Moving Avg (w={window})')
        else:
            ax.plot(self.batch_steps, self.batch_accs, '-',
                    color=COLOR_PALETTE[3], linewidth=2.0, label='Batch Accuracy')

        epoch_starts = []
        seen_epochs = set()
        for step in self.batch_steps:
            epoch_num = step // 1000 + 1
            if epoch_num not in seen_epochs:
                seen_epochs.add(epoch_num)
                epoch_starts.append(step)
        for es in epoch_starts:
            ax.axvline(x=es, color='#AAAAAA', linestyle=':', linewidth=0.8, alpha=0.4)

        ax.set_xlabel('Training Step (Batch)', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{self.experiment_name} \u2014 Batch-Level Training Accuracy', fontsize=14)
        ax.legend(fontsize=10, loc='lower right', frameon=True,
                  fancybox=True, framealpha=0.9)
        ax.set_ylim([0, 1.05])
        plt.tight_layout(pad=1.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path

    def _plot_batch_loss_components(self):
        has_data = any(self.batch_loss_components.values())
        if not has_data:
            return None
        save_path = os.path.join(self.log_dir, f"{self.experiment_name}_batch_loss_components.png")

        fig, ax = plt.subplots(figsize=(16, 6))
        self._add_batch_stage_shading(ax)

        NAMES = ['info_nce', 'occ', 'covariance', 'empathy_ortho', 'rl_policy', 'cls']
        for idx, name in enumerate(NAMES):
            values = self.batch_loss_components[name]
            if values:
                steps = self.batch_steps[:len(values)]
                ax.plot(steps, values, '-', label=name.upper(),
                        linewidth=1.5, alpha=0.85,
                        color=LOSS_COMPONENT_COLORS.get(name))

        ax.set_xlabel('Training Step (Batch)', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title(f'{self.experiment_name} \u2014 Batch-Level Loss Components', fontsize=14)
        ax.legend(fontsize=9, loc='best', ncol=3, frameon=True,
                  fancybox=True, framealpha=0.9)
        plt.tight_layout(pad=1.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path

    def _plot_batch_routing_weights(self):
        if not self.batch_routing_weights:
            return None
        save_path = os.path.join(self.log_dir, f"{self.experiment_name}_batch_routing_weights.png")

        fig, ax = plt.subplots(figsize=(16, 6))
        self._add_batch_stage_shading(ax)

        gnn = [rw.get('gnn', 0) for rw in self.batch_routing_weights]
        llm = [rw.get('llm', 0) for rw in self.batch_routing_weights]
        vis = [rw.get('vis', 0) for rw in self.batch_routing_weights]
        steps = self.batch_steps[:len(self.batch_routing_weights)]

        ax.stackplot(steps, gnn, llm, vis,
                     labels=ROUTING_LABELS,
                     colors=ROUTING_COLORS, alpha=0.7,
                     edgecolor='white', linewidth=0.3)

        ax.set_xlabel('Training Step (Batch)', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_title(f'{self.experiment_name} \u2014 Batch-Level Expert Routing Weights', fontsize=14)
        ax.legend(loc='upper right', fontsize=10, frameon=True,
                  fancybox=True, framealpha=0.9)
        ax.set_ylim([0, 1])
        plt.tight_layout(pad=1.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path

    def _plot_loss_curves(self):
        if not self.train_losses:
            return None
        save_path = os.path.join(self.log_dir, f"{self.experiment_name}_loss_curves.png")

        best_epoch, best_f1, _ = self._get_best_f1_info()

        train_vals = np.array(self.train_losses)
        val_vals = np.array(self.val_losses) if self.val_losses else None

        use_dual_axis = False
        if val_vals is not None and len(val_vals) > 0:
            t_range = train_vals.max() - train_vals.min()
            v_range = val_vals.max() - val_vals.min()
            if t_range > 0 and v_range > 0:
                ratio = v_range / t_range
                if ratio > 2.5 or ratio < 0.4:
                    use_dual_axis = True

        fig, ax1 = plt.subplots(figsize=(10, 6))
        self._add_stage_shading(ax1)

        ax1.plot(self.epochs, self.train_losses, '-o',
                 color=COLOR_PALETTE[0], label='Train Loss',
                 linewidth=2.0, markersize=5, markerfacecolor='white',
                 markeredgewidth=1.5)

        if val_vals is not None and len(val_vals) > 0:
            val_epochs = self.epochs[:len(val_vals)]
            if use_dual_axis:
                ax2 = ax1.twinx()
                ax2.plot(val_epochs, val_vals, '-s',
                         color=COLOR_PALETTE[1], label='Val Loss',
                         linewidth=2.0, markersize=5, markerfacecolor='white',
                         markeredgewidth=1.5)
                ax1.set_ylabel('Train Loss', fontsize=12, color=COLOR_PALETTE[0])
                ax2.set_ylabel('Val Loss', fontsize=12, color=COLOR_PALETTE[1])
                ax1.tick_params(axis='y', labelcolor=COLOR_PALETTE[0])
                ax2.tick_params(axis='y', labelcolor=COLOR_PALETTE[1])

                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                existing_legend = ax1.get_legend()
                if existing_legend:
                    extra_lines = existing_legend.legend_handles
                    extra_labels = [t.get_text() for t in existing_legend.get_texts()]
                    lines2 += extra_lines
                    labels2 += extra_labels
                ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
                           loc='upper left', frameon=True,
                           fancybox=True, framealpha=0.9)
            else:
                ax1.plot(val_epochs, val_vals, '-s',
                         color=COLOR_PALETTE[1], label='Val Loss',
                         linewidth=2.0, markersize=5, markerfacecolor='white',
                         markeredgewidth=1.5)
                ax1.set_ylabel('Loss', fontsize=12)

        if best_epoch is not None and best_f1 is not None:
            ylim = ax1.get_ylim()
            y_annot = ylim[0] + (ylim[1] - ylim[0]) * 0.90
            self._annotate_best_f1(ax1, best_epoch, best_f1, y_pos=y_annot)

        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_title(f'{self.experiment_name} \u2014 Epoch-Level Loss Curves', fontsize=14)
        plt.tight_layout(pad=1.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path

    def _plot_accuracy_curves(self):
        if not self.train_accuracies and not self.val_accuracies:
            return None
        save_path = os.path.join(self.log_dir, f"{self.experiment_name}_accuracy_curves.png")

        best_epoch, best_f1, _ = self._get_best_f1_info()

        fig, ax = plt.subplots(figsize=(10, 6))
        self._add_stage_shading(ax)

        if self.train_accuracies:
            ax.plot(self.epochs[:len(self.train_accuracies)], self.train_accuracies,
                    '-o', color=COLOR_PALETTE[0], label='Train Accuracy',
                    linewidth=2.0, markersize=5, markerfacecolor='white',
                    markeredgewidth=1.5)
        if self.val_accuracies:
            ax.plot(self.epochs[:len(self.val_accuracies)], self.val_accuracies,
                    '-s', color=COLOR_PALETTE[1], label='Val Accuracy',
                    linewidth=2.0, markersize=5, markerfacecolor='white',
                    markeredgewidth=1.5)

        if best_epoch is not None and best_f1 is not None:
            self._annotate_best_f1(ax, best_epoch, best_f1)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(f'{self.experiment_name} \u2014 Epoch-Level Accuracy Curves', fontsize=14)
        ax.legend(fontsize=9, loc='lower right', frameon=True,
                  fancybox=True, framealpha=0.9)
        ax.set_ylim([0, 1.05])
        plt.tight_layout(pad=1.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path

    def _plot_metrics_curves(self):
        n_plots = sum([
            bool(self.val_f1_scores), bool(self.val_auc_scores),
            bool(self.val_precisions), bool(self.val_recalls),
            bool(self.learning_rates)
        ])
        if n_plots == 0:
            return None
        save_path = os.path.join(self.log_dir, f"{self.experiment_name}_metrics_curves.png")

        best_epoch, best_f1, best_idx = self._get_best_f1_info()

        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        idx = 0
        if self.val_f1_scores:
            self._add_stage_shading(axes[idx])
            f1_epochs = self.epochs[:len(self.val_f1_scores)]
            axes[idx].plot(f1_epochs, self.val_f1_scores, '-o',
                           color=COLOR_PALETTE[3], linewidth=2.0, markersize=5,
                           markerfacecolor='white', markeredgewidth=1.5)
            axes[idx].set_title('F1 Score', fontsize=13)
            axes[idx].set_ylabel('F1', fontsize=11)
            axes[idx].set_xlabel('Epoch', fontsize=11)
            axes[idx].set_ylim([0, 1.05])
            if best_epoch is not None and best_f1 is not None:
                axes[idx].scatter([best_epoch], [best_f1],
                                  color='#E74C3C', s=80, zorder=10,
                                  edgecolors='white', linewidth=1.0)
                axes[idx].annotate(
                    f'Best: {best_f1:.3f}',
                    xy=(best_epoch, best_f1),
                    xytext=(0, 12),
                    textcoords='offset points',
                    fontsize=8,
                    color='#E74C3C',
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='#E74C3C', alpha=0.85, linewidth=0.8)
                )
            idx += 1
        if self.val_auc_scores:
            self._add_stage_shading(axes[idx])
            auc_epochs = self.epochs[:len(self.val_auc_scores)]
            axes[idx].plot(auc_epochs, self.val_auc_scores, '-o',
                           color=COLOR_PALETTE[4], linewidth=2.0, markersize=5,
                           markerfacecolor='white', markeredgewidth=1.5)
            axes[idx].set_title('AUC Score', fontsize=13)
            axes[idx].set_ylabel('AUC', fontsize=11)
            axes[idx].set_xlabel('Epoch', fontsize=11)
            axes[idx].set_ylim([0, 1.05])
            idx += 1
        if self.val_precisions:
            self._add_stage_shading(axes[idx])
            prec_epochs = self.epochs[:len(self.val_precisions)]
            axes[idx].plot(prec_epochs, self.val_precisions, '-o',
                           color=COLOR_PALETTE[5], linewidth=2.0, markersize=5,
                           markerfacecolor='white', markeredgewidth=1.5)
            axes[idx].set_title('Precision', fontsize=13)
            axes[idx].set_ylabel('Precision', fontsize=11)
            axes[idx].set_xlabel('Epoch', fontsize=11)
            axes[idx].set_ylim([0, 1.05])
            idx += 1
        if self.val_recalls:
            self._add_stage_shading(axes[idx])
            rec_epochs = self.epochs[:len(self.val_recalls)]
            axes[idx].plot(rec_epochs, self.val_recalls, '-o',
                           color=COLOR_PALETTE[2], linewidth=2.0, markersize=5,
                           markerfacecolor='white', markeredgewidth=1.5)
            axes[idx].set_title('Recall', fontsize=13)
            axes[idx].set_ylabel('Recall', fontsize=11)
            axes[idx].set_xlabel('Epoch', fontsize=11)
            axes[idx].set_ylim([0, 1.05])
            idx += 1
        if self.learning_rates:
            self._add_stage_shading(axes[idx])
            lr_epochs = self.epochs[:len(self.learning_rates)]
            axes[idx].plot(lr_epochs, self.learning_rates, '-o',
                           color=COLOR_PALETTE[0], linewidth=2.0, markersize=5,
                           markerfacecolor='white', markeredgewidth=1.5)
            axes[idx].set_title('Learning Rate', fontsize=13)
            axes[idx].set_ylabel('LR', fontsize=11)
            axes[idx].set_xlabel('Epoch', fontsize=11)
            idx += 1

        plt.suptitle(f'{self.experiment_name} \u2014 Epoch-Level Training Metrics',
                     fontsize=14, y=1.02)
        plt.tight_layout(pad=1.5)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path

    def _plot_loss_components(self):
        has_data = any(self.loss_components.values())
        if not has_data:
            return None
        save_path = os.path.join(self.log_dir, f"{self.experiment_name}_loss_components.png")

        fig, ax = plt.subplots(figsize=(12, 6))
        NAMES = ['info_nce', 'occ', 'covariance', 'empathy_ortho', 'rl_policy', 'cls']
        for name in NAMES:
            values = self.loss_components[name]
            if values:
                ax.plot(range(1, len(values) + 1), values, '-o',
                        label=name.upper(),
                        linewidth=2.0, markersize=5,
                        markerfacecolor='white', markeredgewidth=1.5,
                        color=LOSS_COMPONENT_COLORS.get(name))

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title(f'{self.experiment_name} \u2014 Epoch-Level Loss Components', fontsize=14)
        ax.legend(fontsize=9, loc='best', ncol=3, frameon=True,
                  fancybox=True, framealpha=0.9)
        plt.tight_layout(pad=1.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path

    def _plot_routing_weights(self):
        if not self.routing_weights_history:
            return None
        save_path = os.path.join(self.log_dir, f"{self.experiment_name}_routing_weights.png")

        fig, ax = plt.subplots(figsize=(12, 6))
        gnn_weights = [rw.get('gnn', 0) for rw in self.routing_weights_history]
        llm_weights = [rw.get('llm', 0) for rw in self.routing_weights_history]
        vis_weights = [rw.get('vis', 0) for rw in self.routing_weights_history]
        steps = range(1, len(self.routing_weights_history) + 1)

        ax.stackplot(steps, gnn_weights, llm_weights, vis_weights,
                     labels=ROUTING_LABELS,
                     colors=ROUTING_COLORS, alpha=0.7,
                     edgecolor='white', linewidth=0.3)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Weight', fontsize=12)
        ax.set_title(f'{self.experiment_name} \u2014 Epoch-Level Expert Routing Weights', fontsize=14)
        ax.legend(loc='upper right', fontsize=10, frameon=True,
                  fancybox=True, framealpha=0.9)
        ax.set_ylim([0, 1])
        plt.tight_layout(pad=1.0)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return save_path

    def save_log(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"{self.experiment_name}_log.json")

        log_data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'epochs': self.epochs,
            'stages': self.stages,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'val_precisions': self.val_precisions,
            'val_recalls': self.val_recalls,
            'val_f1_scores': self.val_f1_scores,
            'val_auc_scores': self.val_auc_scores,
            'learning_rates': self.learning_rates,
            'loss_components': self.loss_components,
            'routing_weights_history': self.routing_weights_history,
            'batch_steps': self.batch_steps,
            'batch_losses': self.batch_losses,
            'batch_accs': self.batch_accs,
            'batch_stages': self.batch_stages,
            'batch_loss_components': self.batch_loss_components,
            'batch_routing_weights': self.batch_routing_weights
        }

        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)

        return filepath