# 🌱 DetectFarm — Satellite Imagery Farmland Analyzer

> Computer Vision + Unsupervised ML + Gemini AI Advisory · Built at MANIT Bhopal Research Internship

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-orange)](https://huggingface.co/spaces/Savree97/detectfarm)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green)](https://flask.palletsprojects.com)

---

## What It Does

DetectFarm analyzes satellite or aerial farmland images and automatically:

1. **Detects and segments** individual field plots using OpenCV and scikit-image
2. **Extracts 4 real geometric features** per plot (area, circularity, brightness, solidity)
3. **Clusters plots** into land-use categories using K-Means unsupervised learning
4. **Generates a natural language advisory report** via Gemini 1.5 Flash based on the computed metrics
5. **Exports** an Excel report with per-plot data and advisory recommendations

All features are computed from the actual image — no placeholder or random values.

---

## Screenshots

| Input Image | Detected Boundaries |
|-------------|---------------------|
| *(satellite image)* | *(annotated with plot IDs)* |

| K-Means Clustering (PCA) | Summary Dashboard |
|--------------------------|-------------------|
| *(cluster plot)* | *(stats + advisory)* |

> Upload a satellite image to try it live → [**Live Demo**](https://huggingface.co/spaces/Savree97/detectfarm)

---

## How It Works — Technical Pipeline

```
Input Image (PNG/JPG)
        │
        ▼
   Resize to 512×512
        │
        ▼
   Grayscale + Gaussian Blur (noise reduction)
        │
        ├──► Canny Edge Detection (contour overlay visualization)
        │
        └──► Otsu Thresholding (automatic binary segmentation)
                  │
                  ▼
          Morphological Closing (fill boundary gaps)
                  │
                  ▼
         skimage.label() + regionprops()
         (extract per-region: area, perimeter,
          eccentricity, solidity, intensity_mean)
                  │
                  ▼
         Feature Engineering (4 real features):
         • Area (px²)
         • Circularity = 4π·area / perimeter²
         • Mean Brightness (intensity_mean from regionprops)
         • Solidity = area / convex_hull_area
                  │
                  ▼
         StandardScaler → K-Means (k=3)
         Cluster names assigned by centroid area:
         • Smallest area centroid → Fallow-like
         • Medium area centroid  → Irregular or Small
         • Largest area centroid → Large & Fertile
                  │
                  ▼
         PCA (2D visualization of 4D feature space)
                  │
                  ▼
         Gemini API (summary stats → natural language advisory)
                  │
                  ▼
         Flask renders: annotated images + charts + advisory + Excel
```

---

## Feature Engineering Details

| Feature | Source | Real-world meaning |
|---------|--------|--------------------|
| **Area (px²)** | `region.area` | Plot size — large area = significant land |
| **Circularity** | `4π·area/perimeter²` | Shape regularity (1.0 = circle, <0.1 = jagged) |
| **Mean Brightness** | `region.intensity_mean` | Vegetation density proxy (dark = dense crop / shadow) |
| **Solidity** | `region.solidity` | Shape compactness (1.0 = compact, <0.7 = fragmented) |

All features are computed directly from image pixel data using `skimage.regionprops`.

---

## Advisory Logic

| Signal | Threshold | Advisory |
|--------|-----------|----------|
| Circularity < 0.05 | Highly irregular shape | "Not ideal for drip irrigation" |
| Mean Brightness < 0.4 | Dark / shadowed region | "Lower solar potential" |
| Solidity < 0.7 | Jagged, fragmented boundary | "Limited machinery access" |
| Area < 100 px² | Very small detected region | "Fallow / composting candidate" |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web framework | Flask 3.0 |
| Image processing | OpenCV, Pillow |
| Feature extraction | scikit-image (`regionprops`) |
| ML clustering | scikit-learn (KMeans, StandardScaler, PCA) |
| Data | pandas, NumPy |
| Visualization | Matplotlib |
| AI advisory | Google Gemini 1.5 Flash API |
| Deployment | Hugging Face Spaces (Docker) |

---

## Run Locally

```bash
# Clone
git clone https://github.com/Savree97/DetectFarm-Automated_Farmland_Analyzer.git
cd DetectFarm-Automated_Farmland_Analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Set Gemini API key (optional — app works without it, skips advisory)
export GEMINI_API_KEY=your_key_here   # macOS/Linux
set GEMINI_API_KEY=your_key_here      # Windows

# Run
python app.py
# Open http://localhost:5000
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com) — no credit card required.

---

## Project Context

Built during a Summer Research Internship at **MANIT Bhopal — Centre of Excellence in Product Design and Smart Manufacturing** (June–July 2025).

Research goal: automate farmland plot characterization from low-cost RGB satellite imagery without ground-truth labels, using unsupervised learning.

---

## Author

**Savree Dohar** — B.Tech CSE, Thapar Institute of Engineering and Technology  
[GitHub](https://github.com/Savree97) · [LinkedIn](https://linkedin.com/in/savree-dohar)
