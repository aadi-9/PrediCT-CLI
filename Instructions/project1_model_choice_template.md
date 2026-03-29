# Project 1 — Model Choice Justification (1 paragraph template)

Write 1 paragraph (5–8 sentences) answering:

- What model did you choose (e.g., lightweight 3D U-Net), and what is it predicting (coarse heart mask / bounding box)?
- Why is this model appropriate for COCA cardiac CT (3D context, robust to variable anatomy, works with limited labels)?
- What preprocessing assumptions are you making (HU window, target spacing, input size/cropping)?
- Why should it be faster than TotalSegmentator (smaller architecture, fewer classes, lower resolution, single forward pass)?
- How do you evaluate it (Dice vs TotalSegmentator heart mask, inference time comparison)?
