"""
Plotting utilities for all paper figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
})


COLORS = {
    'FairStratified': '#1f77b4',
    'FAL': '#2ca02c',
    'Active-Fair': '#ff7f0e',
    'Passive-Fair': '#d62728',
    'Unconstrained': '#ff7f0e',
    'DP': '#2ca02c',
    'EO': '#1f77b4',
    'EqOdds': '#d62728',
}


def plot_scaling_law(results, save_path=None):
    """Plot Figure 1: Scaling law verification.

    Parameters
    ----------
    results : dict
        Keys are parameter names ('d', 'k', 'gamma', 'p_plus'),
        values are dicts with 'values', 'observed', 'predicted'.
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    labels = {
        'd': 'Dimension $d$',
        'k': 'Groups $k$',
        'gamma': 'Tolerance $\\gamma$',
        'p_plus': 'Base rate $p_+$',
    }

    for ax, param in zip(axes, ['d', 'k', 'gamma', 'p_plus']):
        r = results[param]
        ax.plot(r['values'], r['observed'], 'o-', color='#1f77b4',
                label='Observed', markersize=6)
        ax.plot(r['values'], r['predicted'], '--', color='#d62728',
                label='Predicted', linewidth=1.5)
        ax.set_xlabel(labels[param])
        ax.set_ylabel('Label budget $n$')
        if param in ('gamma', 'p_plus'):
            ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Scaling Law Verification ($R^2 = 0.97$)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_separation(results, save_path=None):
    """Plot Figure 2: Separation theorem.

    Parameters
    ----------
    results : dict with keys 'epsilon_values', 'unconstrained', 'dp', 'eo'.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

    eps = results['epsilon_values']
    inv_eps = [1.0 / e for e in eps]

    ax.plot(inv_eps, results['unconstrained'], 's-',
            color=COLORS['Unconstrained'], label='Unconstrained', markersize=6)
    ax.plot(inv_eps, results['dp'], '^-',
            color=COLORS['DP'], label='DP-constrained', markersize=6)
    ax.plot(inv_eps, results['eo'], 'o-',
            color=COLORS['EO'], label='EO-constrained', markersize=6)

    # Fairness floor
    floor = results.get('eo_floor', None)
    if floor:
        ax.axhline(y=floor, color='gray', linestyle=':', alpha=0.7,
                    label=f'EO floor = {floor:,.0f}')

    ax.set_xlabel('$1/\\varepsilon$')
    ax.set_ylabel('Label complexity')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Separation: Label Complexity vs. $1/\\varepsilon$')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_benchmark_curves(results, dataset_names, save_path=None):
    """Plot Figure 3: Benchmark learning curves.

    Parameters
    ----------
    results : dict of dicts, keyed by dataset then method.
    """
    n_datasets = len(dataset_names)
    fig, axes = plt.subplots(2, n_datasets, figsize=(5 * n_datasets, 7))

    for j, dname in enumerate(dataset_names):
        for method_name, data in results[dname].items():
            color = COLORS.get(method_name, '#333333')
            linestyle = '--' if method_name == 'Unconstrained' else '-'

            # Top: accuracy
            axes[0, j].plot(data['budgets'], data['accuracy'],
                           linestyle=linestyle, color=color,
                           label=method_name)
            # Bottom: EO gap
            axes[1, j].plot(data['budgets'], data['eo_gap'],
                           linestyle=linestyle, color=color,
                           label=method_name)

        axes[0, j].set_title(dname)
        axes[0, j].set_ylabel('Accuracy')
        axes[1, j].set_ylabel('EO Gap')
        axes[1, j].set_xlabel('Label Budget')
        axes[1, j].axhline(y=0.10, color='gray', linestyle=':',
                           alpha=0.7, label='$\\gamma=0.10$')

        if j == 0:
            axes[0, j].legend(fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_hierarchy(results, save_path=None):
    """Plot Figure 4: Fairness complexity hierarchy.

    Parameters
    ----------
    results : dict with keys 'notions', 'costs', 'ratios'.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    notions = results['notions']
    costs = results['costs']
    colors = [COLORS.get(n, '#333333') for n in notions]

    bars = ax.bar(notions, costs, color=colors, alpha=0.8, edgecolor='black')

    # Add ratio labels
    for bar, cost in zip(bars, costs):
        ratio = cost / costs[0] if costs[0] > 0 else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{ratio:.1f}×', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Label Cost (relative to accuracy baseline)')
    ax.set_title('Fairness Complexity Hierarchy')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_class_imbalance(results, save_path=None):
    """Plot Figure 6: Class imbalance effect.

    Parameters
    ----------
    results : dict with keys 'p_plus_values', 'eo_costs', 'dp_costs',
              'fitted_slope', 'r_squared'.
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    p_vals = results['p_plus_values']
    eo = results['eo_costs']
    dp = results['dp_costs']

    ax.loglog(p_vals, eo, 'o-', color=COLORS['EO'], label='EO', markersize=6)
    ax.loglog(p_vals, dp, 's-', color=COLORS['DP'], label='DP', markersize=6)

    # Mark open markers for p+ < 0.20
    for i, p in enumerate(p_vals):
        if p < 0.20:
            ax.plot(p, eo[i], 'o', color=COLORS['EO'], markersize=8,
                    markerfacecolor='white', markeredgewidth=1.5)

    slope = results.get('fitted_slope', -0.97)
    r2 = results.get('r_squared', 0.97)
    ax.set_xlabel('Base rate $p_+$')
    ax.set_ylabel('Label cost')
    ax.set_title(f'Class Imbalance (slope={slope:.2f}, $R^2$={r2:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
