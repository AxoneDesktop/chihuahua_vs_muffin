
# prepare_dataset_chihuahua_muffin.py

"""
Script to prepare a classroom-friendly Chihuahua vs Muffin dataset.

Option 1 (recommended):
Download the dataset manually from Kaggle:
https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification

Then run this script to create a small subset suitable for class.

Resulting structure:

dataset_chihuahua_muffin/
    train/
        chihuahua/
        muffin/
    val/
        chihuahua/
        muffin/
    test/
        chihuahua/
        muffin/
"""

import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("muffin_vs_chihuahua")  # downloaded dataset
DEST_DIR = Path("dataset_chihuahua_muffin")

TRAIN = 400
VAL = 100
TEST = 100

classes = ["chihuahua", "muffin"]

random.seed(42)

for cls in classes:
    images = list((SOURCE_DIR/cls).glob("*"))
    random.shuffle(images)

    splits = {
        "train": images[:TRAIN],
        "val": images[TRAIN:TRAIN+VAL],
        "test": images[TRAIN+VAL:TRAIN+VAL+TEST]
    }

    for split, imgs in splits.items():
        target = DEST_DIR/split/cls
        target.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy(img, target/img.name)

print("Dataset preparado en:", DEST_DIR)
