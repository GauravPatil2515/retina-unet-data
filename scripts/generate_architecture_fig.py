import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, facecolor='#E3F2FD', edgecolor='#1565C0'):
    box = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=11, fontweight='bold', family='sans-serif', zorder=3)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1.5, headwidth=8, shrink=0), zorder=1)

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# Preprocessing Pipeline
draw_box(ax, 0.5, 4.5, 2, 1, "Raw RGB\nFundus Image", facecolor='#F3E5F5', edgecolor='#6A1B9A')
draw_arrow(ax, 2.5, 5.0, 3.0, 5.0)

draw_box(ax, 3.0, 4.5, 2.5, 1, "Preprocessing\n(Green Channel, CLAHE,\nFOV Masking)", facecolor='#E8F5E9', edgecolor='#2E7D32')
draw_arrow(ax, 5.5, 5.0, 6.0, 5.0)

draw_box(ax, 6.0, 4.5, 2, 1, "Patch Extraction\n(128x128)", facecolor='#E8F5E9', edgecolor='#2E7D32')
draw_arrow(ax, 7.0, 4.5, 7.0, 3.5)

# Training Stages
draw_box(ax, 5.5, 2.5, 3, 1, "Curriculum Learning\nStage 1: High Density\nStage 2: Medium Density\nStage 3: Sparse/All", facecolor='#FFF3E0', edgecolor='#E65100')
draw_arrow(ax, 7.0, 2.5, 7.0, 1.5)

# Model Architecture
draw_box(ax, 5.5, 0.5, 3, 1, "U-Net++\nArchitecture", facecolor='#E3F2FD', edgecolor='#1565C0')
draw_arrow(ax, 8.5, 1.0, 9.5, 1.0)
draw_arrow(ax, 8.5, 0.8, 9.5, 0.8)

# Losses
draw_box(ax, 9.5, 1.5, 2.3, 1, "Topological Loss\n(Persistent Homology)", facecolor='#FFEBEE', edgecolor='#C62828')
draw_box(ax, 9.5, 0.2, 2.3, 1, "BCE + Dice\n+ SkelDice", facecolor='#FFEBEE', edgecolor='#C62828')

# Connect model to losses
draw_arrow(ax, 8.5, 1.2, 9.5, 2.0)

plt.title("Retinal Vessel Segmentation: Preprocessing and Architecture Pipeline", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('paper/fig10_architecture.png', dpi=300, bbox_inches='tight')
print("Architecture figure generated at paper/fig10_architecture.png")
