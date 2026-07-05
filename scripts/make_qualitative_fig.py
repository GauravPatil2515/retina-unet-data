from PIL import Image
import numpy as np

# Load images
img_a = Image.open('results/ablation_results/exp_a_baseline/evaluation/prediction_1.png')
img_d = Image.open('results/ablation_results/exp_d_ours/evaluation/prediction_1.png')

# The size is (2218, 765). 3 panels = 2218 / 3 = 739.3
w, h = img_a.size
panel_w = w // 3

# Crop panels
orig = img_d.crop((0, 0, panel_w, h))
gt = img_d.crop((panel_w, 0, 2*panel_w, h))
pred_baseline = img_a.crop((2*panel_w, 0, w, h))
pred_ours = img_d.crop((2*panel_w, 0, w, h))

# Combine horizontally
new_w = panel_w * 4
new_img = Image.new('RGB', (new_w, h))
new_img.paste(orig, (0, 0))
new_img.paste(gt, (panel_w, 0))
new_img.paste(pred_baseline, (2*panel_w, 0))
new_img.paste(pred_ours, (3*panel_w, 0))

# Save
new_img.save('paper/fig11_qualitative.png')
print("Saved paper/fig11_qualitative.png")
