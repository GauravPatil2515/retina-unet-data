"""
Protocol Flowchart Figure
=========================
Creates Figure: GT → Prediction → Skeleton → Graph Extraction → 
Connectivity Analysis → Structural Metrics → Clinical Interpretation
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle
from pathlib import Path

def create_protocol_flowchart():
    fig, ax = plt.subplots(figsize=(12, 2))
    
    steps = [
        "Ground Truth\nMask",
        "Model\nPrediction",
        "Skeleton\nExtraction",
        "Graph\nExtraction",
        "Connectivity\nAnalysis",
        "Structural\nMetrics",
        "Clinical\nInterpretation",
        "Protocol\nReport"
    ]
    
    n_steps = len(steps)
    x_pos = np.linspace(0.1, 0.9, n_steps)
    
    # Draw boxes
    for i, (step, x) in enumerate(zip(steps, x_pos)):
        box = FancyBboxPatch((x-0.08, 0.3), 0.16, 0.4,
                             boxstyle="round,pad=0.02",
                             facecolor='lightblue',
                             edgecolor='darkblue',
                             linewidth=2)
        ax.add_patch(box)
        ax.text(x, 0.5, step, ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Draw arrows
    for i in range(n_steps - 1):
        ax.annotate('', (x_pos[i+1]-0.08, 0.5), (x_pos[i]+0.08, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Beyond Dice: Topology-Aware Evaluation Protocol', 
                 fontsize=14, fontweight='bold', pad=20)
    
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    plt.savefig('results/figures/fig_protocol_flowchart.png', dpi=150, 
                bbox_inches='tight')
    plt.close()
    print("[OK] Protocol flowchart saved")

if __name__ == "__main__":
    import numpy as np
    create_protocol_flowchart()