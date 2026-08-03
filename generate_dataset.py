"""
generate_dataset.py
--------------------
NOTE: This sandbox has no network access to the UCI repository, so this
script SYNTHESIZES a realistic stand-in for the Dry Bean Dataset
(Koklu & Ozkan, 2020) using per-class means/std/correlation structure
consistent with the published paper and community reproductions.

To use the REAL dataset instead (recommended for your actual submission):
1. Download "Dry_Bean_Dataset.xlsx" from:
   https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
   or Kaggle: https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset
2. Place it in this folder.
3. Replace the dataset-loading cell in dry_bean_classification.py with:
       df = pd.read_excel("Dry_Bean_Dataset.xlsx")
   Everything downstream (preprocessing, scaling, ANN, evaluation) is
   already written generically and needs NO other changes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Real published class sizes (Koklu & Ozkan, 2020)
class_sizes = {
    "SEKER": 2027,
    "BARBUNYA": 1322,
    "BOMBAY": 522,
    "CALI": 1630,
    "HOROZ": 1928,
    "SIRA": 2636,
    "DERMASON": 3546,
}

# Approximate per-class centers for key physical measurements, derived from
# the ranges/summary statistics reported in the original paper. Correlated
# features (Perimeter, MajorAxisLength, ConvexArea, etc.) are derived from
# Area + shape ratios rather than sampled independently, so the synthetic
# data preserves the realistic multicollinearity real seed-morphology data has.
area_center = {
    "SEKER": 39600, "BARBUNYA": 69700, "BOMBAY": 173100,
    "CALI": 76500, "HOROZ": 47700, "SIRA": 44200, "DERMASON": 32200,
}
area_spread = {
    "SEKER": 3800, "BARBUNYA": 6800, "BOMBAY": 14500,
    "CALI": 7200, "HOROZ": 4600, "SIRA": 3600, "DERMASON": 2600,
}
aspect_ratio_center = {  # MajorAxis / MinorAxis
    "SEKER": 1.29, "BARBUNYA": 1.55, "BOMBAY": 1.35,
    "CALI": 1.62, "HOROZ": 1.75, "SIRA": 1.42, "DERMASON": 1.47,
}
roundness_center = {  # (4*pi*Area)/Perimeter^2
    "SEKER": 0.90, "BARBUNYA": 0.84, "BOMBAY": 0.88,
    "CALI": 0.82, "HOROZ": 0.80, "SIRA": 0.87, "DERMASON": 0.90,
}

rows = []
for cls, n in class_sizes.items():
    area = np.random.normal(area_center[cls], area_spread[cls], n).clip(min=1000)
    aspect = np.random.normal(aspect_ratio_center[cls], 0.08, n).clip(min=1.02)
    roundness = np.random.normal(roundness_center[cls], 0.03, n).clip(0.5, 0.99)

    # Derive minor/major axis from area (ellipse approx) and aspect ratio
    # Area_ellipse = pi/4 * Major * Minor ; Major = aspect * Minor
    minor_axis = np.sqrt((4 * area) / (np.pi * aspect))
    major_axis = aspect * minor_axis

    # Perimeter from roundness relation: roundness = 4*pi*Area / Perimeter^2
    perimeter = np.sqrt((4 * np.pi * area) / roundness)

    convex_area = area * np.random.normal(1.007, 0.004, n)
    equiv_diameter = np.sqrt(4 * area / np.pi)
    extent = np.random.normal(0.75, 0.03, n).clip(0.5, 0.95)
    solidity = area / convex_area
    compactness = equiv_diameter / major_axis

    eccentricity = np.sqrt(np.clip(1 - (minor_axis**2 / major_axis**2), 0, 0.999))

    shape_factor1 = major_axis / area
    shape_factor2 = minor_axis / area
    shape_factor3 = (area) / (major_axis * minor_axis * np.pi / 4) * compactness
    shape_factor4 = compactness * np.random.normal(1.0, 0.01, n)

    df_cls = pd.DataFrame({
        "Area": area,
        "Perimeter": perimeter,
        "MajorAxisLength": major_axis,
        "MinorAxisLength": minor_axis,
        "AspectRation": aspect,
        "Eccentricity": eccentricity,
        "ConvexArea": convex_area,
        "EquivDiameter": equiv_diameter,
        "Extent": extent,
        "Solidity": solidity,
        "roundness": roundness,
        "Compactness": compactness,
        "ShapeFactor1": shape_factor1,
        "ShapeFactor2": shape_factor2,
        "ShapeFactor3": shape_factor3,
        "ShapeFactor4": shape_factor4,
        "Class": cls,
    })
    rows.append(df_cls)

df = pd.concat(rows, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
df.to_csv("Dry_Bean_Dataset_synthetic.csv", index=False)
print("Saved Dry_Bean_Dataset_synthetic.csv with shape:", df.shape)
print(df["Class"].value_counts())
