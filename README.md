# 🌱 DetectFarm - Automated Farmland Analyzer

**DetectFarm** is a full-stack web application that leverages **computer vision** and **data science** to provide **automated analysis of agricultural land** from satellite or aerial imagery.  
It modernizes farmland management by **identifying individual plots**, **calculating key metrics**, and **generating actionable insights** — all through an intuitive and responsive web interface.

---

## ✨ Features

### 🔍 Automated Plot Detection
- Uses advanced **image processing** to accurately **detect and segment** farmland plots from a single satellite/aerial image.

### 📏 Quantitative Land Analysis
For each detected plot, the system calculates:
- **Total Area** and **Perimeter**
- **Circularity** and **Solidity**
- **Fallow vs. Non-Fallow** status

### 💡 Actionable Insights
- **Fragmentation Index** for overall land utilization.
- **Irrigation advisory** for better water management.

### 📊 Data-Driven Visualizations
Automatically generates plots with **Matplotlib**:
- Histogram of plot areas  
- Box plot of plot circularity  
- Pie chart of fallow vs. non-fallow plots  
- Bar chart of irrigation advisory distribution  

### 🧠 Advanced Clustering
- Uses **K-Means** and **PCA** to group similar plots and reveal hidden patterns.

### 📥 Downloadable Reports
- Export **Excel reports** containing raw data and analysis for all detected plots.

---

## ⚙️ Technology Stack

### **Backend & Data Analysis**
- Python 3.x
- Flask – Web framework
- OpenCV (`cv2`) – Image preprocessing & contour detection
- scikit-image – Feature extraction with `regionprops`
- NumPy – Numerical operations
- Pandas – Data manipulation & report generation
- Matplotlib – Data visualization
- scikit-learn – K-Means clustering & PCA

### **Frontend**
- HTML5
- Bootstrap 5 – Responsive UI

---

## 🚀 How It Works

1. **Upload Image** – User uploads a satellite/aerial farmland image.
2. **Preprocessing** – Image is resized, converted to grayscale, denoised, and processed using Otsu’s thresholding & Canny edge detection.
3. **Contour & Feature Extraction** – Detects farmland boundaries and calculates shape metrics using `regionprops`.
4. **Data Analysis** – Compiles extracted metrics into a Pandas DataFrame, applies K-Means clustering, and generates PCA visualizations.
5. **Visualization & Output** – Renders:
   - Annotated image with detected plots
   - Statistical charts
   - Downloadable Excel report

---

## 🛠️ Installation

### **Prerequisites**
- Python 3.x
- pip (Python package manager)

### **Steps**

```bash
# 1️⃣ Clone the repository
git clone https://github.com/your-username/DetectFarm.git
cd DetectFarm

# 2️⃣ Create a virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3️⃣ Install dependencies
pip install Flask numpy opencv-python scikit-image pandas matplotlib scikit-learn

## ▶️ Usage

```bash
# Run the Flask app
python app.py
Open your browser and go to: http://127.0.0.1:5000

Upload a farmland image using the web interface.

View:

Detected plots with labels

Statistical charts

Downloadable Excel report


👤 Author
Savree Dohar
