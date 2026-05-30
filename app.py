"""
DetectFarm — Satellite Imagery Farmland Analyzer
Author: Savree Dohar
GitHub: https://github.com/Savree97/DetectFarm-Automated_Farmland_Analyzer

CHANGES FROM v1:
- Replaced np.random.uniform() shadow_ratio with real intensity_mean from regionprops
- Replaced np.random.uniform() access_score with real solidity from regionprops
- Fixed K-Means cluster labeling (was hardcoded, now centroid-aware)
- Added Gemini API advisory generation
- Fixed Flask template folder structure (templates/ subfolder)
- Fixed image URL serving (url_for instead of raw filesystem paths)
- Fixed path traversal vulnerability in download route
- Added error handling for empty detection results
- Added file cleanup to prevent disk fill
- Fixed debug=True in production
"""

import os
import uuid
import shutil

from flask import Flask, render_template, request, send_file, url_for, abort
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # MUST be before pyplot import — prevents GUI errors on server
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Gemini setup (optional — works without API key, just skips advisory) ──────
try:
    from google import genai as _google_genai
    GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
    if GEMINI_KEY:
        _genai_client = _google_genai.Client(api_key=GEMINI_KEY)
        GEMINI_AVAILABLE = True
    else:
        GEMINI_AVAILABLE = False
        _genai_client = None
except ImportError:
    try:
        import google.generativeai as _old_genai
        GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
        if GEMINI_KEY:
            _old_genai.configure(api_key=GEMINI_KEY)
            _genai_client = _old_genai.GenerativeModel("gemini-1.5-flash")
            GEMINI_AVAILABLE = True
        else:
            GEMINI_AVAILABLE = False
            _genai_client = None
    except ImportError:
        GEMINI_AVAILABLE = False
        _genai_client = None

# ── Flask app setup ────────────────────────────────────────────────────────────
app = Flask(__name__)

# FIX: Use app.static_folder to build the output path correctly.
# This ensures it works both locally and on deployment platforms.
OUTPUT_FOLDER = os.path.join(app.static_folder, 'output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Maximum files to keep in output folder before cleanup (prevents disk fill)
MAX_OUTPUT_FILES = 50


def cleanup_old_files():
    """Delete oldest files if output folder exceeds MAX_OUTPUT_FILES."""
    files = [
        os.path.join(OUTPUT_FOLDER, f)
        for f in os.listdir(OUTPUT_FOLDER)
        if os.path.isfile(os.path.join(OUTPUT_FOLDER, f))
    ]
    if len(files) > MAX_OUTPUT_FILES:
        # Sort by creation time, delete the oldest half
        files.sort(key=os.path.getctime)
        for f in files[:len(files) // 2]:
            try:
                os.remove(f)
            except OSError:
                pass


def build_static_url(filename):
    """
    Convert a filesystem path to a Flask static URL.
    Uses basename only + 'output/' prefix — works on both Windows and Linux.
    os.path.relpath was broken on Windows (returns backslashes that url_for rejects).
    """
    basename = os.path.basename(filename)
    return url_for('static', filename='output/' + basename)


def get_gemini_advisory(summary_stats: dict, df: pd.DataFrame) -> str:
    """
    Generate a natural language farm advisory using Gemini API.

    HOW IT WORKS:
    1. We extract the key computed statistics (fallow count, cluster breakdown,
       average circularity, land utilization) into a structured summary string.
    2. We send this summary to Gemini with a specific prompt asking for a
       practical 3-paragraph farm advisory.
    3. Gemini generates natural language text grounded in our actual numbers.

    WHY THIS IS LEGITIMATE AI:
    - The input (plot metrics) comes from real image analysis
    - The LLM adds natural language interpretation of numerical data
    - It does NOT make up the numbers — it explains the ones we computed
    - This is the standard "LLM as report generator" pattern used in production
      systems (e.g., Microsoft Copilot for Excel, Salesforce Einstein)

    Returns empty string if Gemini is unavailable (graceful degradation).
    """
    if not GEMINI_AVAILABLE or _genai_client is None or df.empty:
        return ""

    try:
        cluster_breakdown = df['Cluster Type'].value_counts().to_dict()
        irrigation_breakdown = df['Irrigation Advisory'].value_counts().to_dict()

        prompt = f"""You are an agricultural advisor analyzing satellite imagery data.

Farm Analysis Results (computed from computer vision analysis):
- Total plots detected: {len(df)}
- Fallow plots (area < 100px threshold): {int(df['Is Fallow?'].sum())}
- Active plots: {int((~df['Is Fallow?']).sum())}
- Land utilization: {summary_stats.get('Land utilization (%)', 'N/A')}%
- Average plot circularity: {summary_stats.get('Average Circularity', 'N/A')} (1.0 = perfect circle, <0.1 = very irregular shape)
- Average brightness (vegetation proxy): {round(df['Mean Brightness'].mean(), 3)} (0=dark/shadowed, 1=bright/dry)
- Average solidity (shape compactness): {round(df['Solidity'].mean(), 3)} (1.0 = compact, <0.7 = jagged boundary)
- Cluster breakdown: {cluster_breakdown}
- Irrigation suitability: {irrigation_breakdown}
- Fragmentation index: {summary_stats.get('Fragmentation Index', 'N/A')}

Write a concise 3-paragraph advisory report for a smallholder farmer covering:
1. Current land status (what the data shows about plot sizes and shapes)
2. Irrigation recommendations based on plot circularity and the number of drip-suitable plots
3. Priority actions (which plots to focus on first, what to do with fallow land)

Use simple, practical language. Do not repeat the raw numbers verbatim — interpret them.
Keep each paragraph to 2-3 sentences. Do not use bullet points."""

        # Works with both google-genai (new) and google-generativeai (old)
        if hasattr(_genai_client, 'models'):
            # New google-genai SDK (AQ. keys)
            response = _genai_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
        else:
            # Old google-generativeai SDK (AIza keys)
            response = _genai_client.generate_content(prompt)
        return response.text

    except Exception as e:
        # Graceful degradation — if Gemini fails, app still works
        return f"[Advisory generation unavailable: {str(e)}]"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('index.html', error="No file uploaded.")

        cleanup_old_files()

        uid = str(uuid.uuid4())
        filename = uid + '.png'
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        file.save(filepath)

        # ── IMAGE PREPROCESSING ────────────────────────────────────────────────
        # Step 1: Load image and convert to RGB (handles JPG, PNG, etc.)
        image = Image.open(filepath).convert('RGB')
        image_np = np.array(image)

        # Step 2: Resize to fixed 512x512
        # WHY: regionprops features (area, perimeter) are pixel-based.
        # Fixing resolution makes them comparable across different input images.
        resized_img = cv2.resize(image_np, (512, 512))

        # Step 3: Convert to grayscale for thresholding
        gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)

        # Step 4: Normalize grayscale to 0.0-1.0 float
        # WHY: regionprops intensity_mean returns the mean of the intensity_image
        # we pass in. By normalizing, we get a 0-1 brightness score per region.
        gray_normalized = gray.astype(float) / 255.0

        # Step 5: Gaussian blur to reduce noise before edge/threshold operations
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 6: Canny edge detection for contour overlay visualization
        edges = cv2.Canny(blurred, 50, 150)

        # Step 7: Otsu's thresholding
        # WHY OTSU: Instead of manually setting a threshold (e.g., > 128 = white),
        # Otsu automatically finds the value that minimizes intra-class variance.
        # Works well when the image has two dominant intensity groups (soil vs crop).
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 8: Morphological closing
        # WHY: Closing = dilation then erosion. It fills small holes/gaps in the
        # binary mask. Field boundaries often have small breaks from noise or shadows.
        # Closing bridges those gaps without changing overall shape.
        try:
            from skimage.morphology import footprint_rectangle
            binary_closed = closing(binary > 0, footprint_rectangle((3, 3)))
        except ImportError:
            binary_closed = closing(binary > 0, square(3))

        # Step 9: Label connected components
        # WHY: label() assigns a unique integer ID to each connected white region.
        # Each labeled region = one detected farmland plot candidate.
        label_img = label(binary_closed)

        # Step 10: Extract region properties
        # CRITICAL FIX: Pass gray_normalized as intensity_image so regionprops
        # can compute intensity_mean — the actual pixel brightness of each region.
        # This REPLACES the old np.random.uniform() shadow_ratio.
        regions = regionprops(label_img, intensity_image=gray_normalized)

        # ── CONTOUR OVERLAY ────────────────────────────────────────────────────
        contour_img = resized_img.copy()
        cv2_contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(contour_img, cv2_contours, -1, (0, 255, 0), 1)

        # ── FEATURE EXTRACTION ─────────────────────────────────────────────────
        plot_data = []
        used_coords = []
        min_distance = 7  # minimum pixel distance between plot centroids
        total_area = binary_closed.size
        used_area = 0

        for idx, region in enumerate(regions):
            area = region.area
            perimeter = region.perimeter

            # Filter out noise regions (too small to be real plots)
            if area < 50 or perimeter < 20:
                continue

            # ── REAL FEATURES (all computed from image data) ───────────────────

            # Feature 1: Circularity
            # Formula: 4π·area / perimeter²
            # Meaning: How close to a circle the region is.
            # 1.0 = perfect circle. < 0.1 = very jagged/irregular boundary.
            # Real-world meaning: Regular shaped plots are easier to irrigate uniformly.
            circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

            # Feature 2: Mean Brightness (replaces shadow_ratio random)
            # SOURCE: regionprops intensity_mean — actual pixel brightness of this region
            # Range: 0.0 (dark/shadowed/dense vegetation) to 1.0 (bright/dry/bare soil)
            # WHY THIS MATTERS FOR SOLAR: Darker regions = more vegetation or shadow
            # cover = lower solar potential. Brighter = more open = better solar.
            # This reads REAL data from the image. Not random.
            mean_brightness = round(region.intensity_mean, 4)

            # Feature 3: Solidity (replaces access_score random)
            # Formula: area / convex_hull_area
            # Range: 0.0 to 1.0 (always ≤ 1.0)
            # 1.0 = perfectly convex, compact region (easy machinery access)
            # < 0.7 = jagged, concave boundary (harder to access, irregular plot)
            # WHY THIS MATTERS: Irregular plot boundaries suggest fragmented land
            # or natural obstacles — a real proxy for mechanical accessibility.
            # This reads REAL shape data. Not random.
            solidity = round(region.solidity, 4)

            # ── ADVISORY LOGIC (all based on real computed features) ───────────

            # Fallow detection: small area = likely a fragmented/unused patch
            is_fallow = bool(area < 100)

            # Irrigation: irregular shapes (low circularity) = uneven water distribution
            irrigation = "Not ideal for drip" if circularity < 0.05 else "Drip suitable"

            # Solar: based on actual image brightness of the region
            # Low brightness = darker = more shade/vegetation = lower solar potential
            solar = "Lower solar potential" if mean_brightness < 0.4 else "Good for solar panels"

            # Access: based on actual shape compactness (solidity)
            access = "Limited machinery access" if solidity < 0.7 else "Good machinery access"

            # Land reuse suggestion
            reuse = "Apiary or composting" if is_fallow else "Continue cropping"

            priority = 1 if is_fallow else 0
            final = f"{reuse} | {irrigation} | {solar} | {access}"

            # Deduplicate overlapping centroids
            y, x = region.centroid
            if any(
                np.linalg.norm(np.array([x, y]) - np.array(c)) < min_distance
                for c in used_coords
            ):
                continue
            used_coords.append((x, y))

            # Annotate plot ID on contour image
            plot_id = idx + 100
            cx, cy = int(x), int(y)
            cv2.putText(contour_img, str(plot_id), (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 3,
                        lineType=cv2.LINE_AA)
            cv2.putText(contour_img, str(plot_id), (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
                        lineType=cv2.LINE_AA)

            plot_data.append({
                'Plot ID': plot_id,
                'Area (px²)': round(area, 2),
                'Perimeter': round(perimeter, 2),
                'Circularity': round(circularity, 6),
                'Mean Brightness': mean_brightness,   # was: Shadow Ratio (random)
                'Solidity': solidity,                  # was: Access Score (random)
                'Is Fallow?': is_fallow,
                'Reuse Suggestion': reuse,
                'Irrigation Advisory': irrigation,
                'Solar Advisory': solar,
                'Machinery Access': access,
                'Priority Score': priority,
                'Final Suggestion': final
            })
            used_area += area

        df = pd.DataFrame(plot_data)

        # ── EMPTY DETECTION GUARD ──────────────────────────────────────────────
        # FIX: Old code would crash here if no plots were detected (empty DataFrame).
        # df['Area (px²)'].mean() on empty df returns NaN, then summary_stats crashes.
        if df.empty:
            os.remove(filepath)
            return render_template(
                'index.html',
                error="No farmland plots detected. Try a clearer satellite image "
                      "with visible field boundaries."
            )

        # ── SUMMARY STATISTICS ─────────────────────────────────────────────────
        utilization = (used_area / total_area) * 100
        fragmentation_index = len(df) / total_area

        summary_stats = {
            "Total image area (pixels)": total_area,
            "Detected plot area (pixels)": round(used_area, 2),
            "Land utilization (%)": round(utilization, 2),
            "Fragmentation Index": round(fragmentation_index, 6),
            "Average plot area (px²)": round(df['Area (px²)'].mean(), 2),
            "Average Circularity": round(df['Circularity'].mean(), 4),
            "Average Brightness": round(df['Mean Brightness'].mean(), 4),
            "Average Solidity": round(df['Solidity'].mean(), 4),
            "Fallow plots": int(df['Is Fallow?'].sum()),
            "Active plots": int((~df['Is Fallow?']).sum()),
            "Total plots detected": len(df),
        }

        # ── K-MEANS CLUSTERING ────────────────────────────────────────────────
        feature_cols = ['Area (px²)', 'Circularity', 'Mean Brightness', 'Solidity']
        features = df[feature_cols]
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)

        n_plots = len(df)
        n_clusters = min(3, n_plots)
        all_names = ["Fallow-like", "Irregular or Small", "Large & Fertile"]

        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['Cluster'] = kmeans.fit_predict(scaled_features)
            centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)
            area_col_idx = feature_cols.index('Area (px²)')
            sorted_by_area = np.argsort(centers_original_scale[:, area_col_idx])
            cluster_name_map = {
                int(sorted_by_area[i]): all_names[i]
                for i in range(n_clusters)
            }
            df['Cluster Type'] = df['Cluster'].map(cluster_name_map)
        else:
            # Only 1 plot
            df['Cluster'] = 0
            df['Cluster Type'] = df['Area (px²)'].apply(
                lambda a: "Fallow-like" if a < 100 else "Large & Fertile"
            )
            cluster_name_map = {0: df['Cluster Type'].iloc[0]}

        # ── PCA VISUALIZATION ─────────────────────────────────────────────────
        n_pca = min(2, n_plots, len(feature_cols))
        pca = PCA(n_components=n_pca)
        pca_components_raw = pca.fit_transform(scaled_features)
        # Always ensure 2 columns and 2 explained values for consistent plotting
        if n_pca == 1:
            pca_components = np.column_stack(
                [pca_components_raw, np.zeros(len(pca_components_raw))]
            )
            explained = [float(pca.explained_variance_ratio_[0]), 0.0]
        else:
            pca_components = pca_components_raw
            explained = [float(pca.explained_variance_ratio_[0]),
                         float(pca.explained_variance_ratio_[1])]

        plt.figure(figsize=(7, 5))
        scatter = plt.scatter(
            pca_components[:, 0], pca_components[:, 1],
            c=df['Cluster'], cmap='tab10', s=60, edgecolor='k', alpha=0.8
        )
        plt.title("K-Means Clustering of Plot Features (PCA Projection)")
        plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
        plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")

        handles = []
        for cluster_id, name in cluster_name_map.items():
            color = plt.cm.tab10(cluster_id / 10)
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=color, markersize=8, label=name))
        plt.legend(handles=handles, title="Cluster Type")
        plt.colorbar(scatter, label="Cluster ID")
        cluster_plot_path = os.path.join(OUTPUT_FOLDER, f"cluster_plot_{uid}.png")
        plt.tight_layout()
        plt.savefig(cluster_plot_path)
        plt.close()

        # ── EXCEL EXPORT ───────────────────────────────────────────────────────
        excel_path = os.path.join(OUTPUT_FOLDER, f"advisory_{uid}.xlsx")
        df.to_excel(excel_path, index=False)
        table_html = df.head(10).to_html(classes='table table-striped table-sm', index=False)

        # ── SAVE IMAGES ────────────────────────────────────────────────────────
        original_path = os.path.join(OUTPUT_FOLDER, f'original_{filename}')
        contour_path = os.path.join(OUTPUT_FOLDER, f'contour_{filename}')
        Image.fromarray(resized_img).save(original_path)
        Image.fromarray(contour_img).save(contour_path)

        # ── MATPLOTLIB CHARTS ──────────────────────────────────────────────────
        # Chart 1: Area distribution histogram
        plt.figure(figsize=(6, 4))
        plt.hist(df['Area (px²)'], bins=min(15, len(df)), color='skyblue', edgecolor='black')
        plt.title("Distribution of Plot Areas")
        plt.xlabel("Area (px²)")
        plt.ylabel("Count")
        hist_path = os.path.join(OUTPUT_FOLDER, f"hist_area_{uid}.png")
        plt.tight_layout()
        plt.savefig(hist_path)
        plt.close()

        # Chart 2: Circularity boxplot
        plt.figure(figsize=(6, 4))
        plt.boxplot(df['Circularity'], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'))
        plt.title("Boxplot of Plot Circularity")
        plt.ylabel("Circularity (1.0 = perfect circle)")
        plt.ylim(0, max(1.5, df['Circularity'].max() * 1.1))
        boxplot_path = os.path.join(OUTPUT_FOLDER, f"boxplot_{uid}.png")
        plt.tight_layout()
        plt.savefig(boxplot_path)
        plt.close()

        # Chart 3: Fallow vs Active pie
        plt.figure(figsize=(5, 5))
        fallow_counts = df['Is Fallow?'].value_counts()
        pie_labels = []
        if True in fallow_counts.index:
            pie_labels.append(f"Fallow ({int(fallow_counts.get(True, 0))})")
        if False in fallow_counts.index:
            pie_labels.append(f"Active ({int(fallow_counts.get(False, 0))})")
        plt.pie(fallow_counts, labels=pie_labels, autopct='%1.1f%%',
                startangle=140, colors=['#ff9999', '#66b3ff'])
        plt.title("Fallow vs Active Plots")
        pie_path = os.path.join(OUTPUT_FOLDER, f"fallow_pie_{uid}.png")
        plt.tight_layout()
        plt.savefig(pie_path)
        plt.close()

        # Chart 4: Irrigation advisory bar
        plt.figure(figsize=(6, 4))
        df['Irrigation Advisory'].value_counts().plot(
            kind='bar', color='lightgreen', edgecolor='black'
        )
        plt.title("Irrigation Advisory Distribution")
        plt.xlabel("Advisory")
        plt.ylabel("Number of Plots")
        plt.xticks(rotation=15, ha='right')
        barplot_path = os.path.join(OUTPUT_FOLDER, f"irrigation_bar_{uid}.png")
        plt.tight_layout()
        plt.savefig(barplot_path)
        plt.close()

        # Chart 5: Brightness distribution (new — shows the real image data)
        plt.figure(figsize=(6, 4))
        plt.hist(df['Mean Brightness'], bins=min(15, len(df)),
                 color='#ffd700', edgecolor='black')
        plt.title("Distribution of Plot Brightness\n(proxy for vegetation density)")
        plt.xlabel("Mean Brightness (0=dark/shadowed, 1=bright/dry)")
        plt.ylabel("Count")
        brightness_path = os.path.join(OUTPUT_FOLDER, f"brightness_{uid}.png")
        plt.tight_layout()
        plt.savefig(brightness_path)
        plt.close()

        # ── GEMINI ADVISORY ────────────────────────────────────────────────────
        advisory_text = get_gemini_advisory(summary_stats, df)

        # ── FIX: Convert filesystem paths to Flask static URLs ─────────────────
        # Old code passed raw paths like 'static/output/img.png' to the template.
        # These are filesystem paths, not HTTP URLs. They work locally by accident
        # but break on any deployment. url_for() generates the correct URL always.
        def to_url(path):
            return build_static_url(path)

        return render_template(
            'index.html',
            original=to_url(original_path),
            contour=to_url(contour_path),
            summary=summary_stats,
            table_html=table_html,
            excel_filename=f"advisory_{uid}.xlsx",  # filename only, not full path
            plots={
                'area_hist': to_url(hist_path),
                'boxplot': to_url(boxplot_path),
                'fallow_pie': to_url(pie_path),
                'irrigation_bar': to_url(barplot_path),
                'cluster_plot': to_url(cluster_plot_path),
                'brightness': to_url(brightness_path),
            },
            advisory=advisory_text,
            gemini_available=GEMINI_AVAILABLE,
        )

    return render_template('index.html')


@app.route('/download/<filename>')
def download(filename):
    """
    Serve Excel reports for download.

    FIX: Old route used <path:filename> which accepted any filesystem path —
    a path traversal vulnerability. A user could request /download/../app.py
    to download your source code.

    Fix: Accept only a filename (no slashes), validate it exists in OUTPUT_FOLDER,
    then serve it from there. Users can never escape the output folder.
    """
    # Reject any path components (slashes)
    if '/' in filename or '\\' in filename or '..' in filename:
        abort(400)

    safe_path = os.path.join(OUTPUT_FOLDER, filename)

    # Verify the file actually exists in our output folder
    if not os.path.isfile(safe_path):
        abort(404)

    return send_file(safe_path, as_attachment=True, download_name=filename)


if __name__ == '__main__':
    # FIX: debug=True was set unconditionally in the original.
    # In production, debug=True exposes an interactive debugger (security risk)
    # and disables performance optimizations.
    # Use the DEBUG environment variable to control this.
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
