DetectFarm - Automated Farmland Analyzer
🌟 Project Overview
DetectFarm is a robust web application that leverages computer vision and data science to provide automated analysis of agricultural land from satellite or aerial imagery. This tool modernizes farmland management by quickly identifying individual plots, calculating key metrics, and generating actionable insights through a user-friendly interface.

This project was built to showcase a full-stack application that combines Flask for the backend, advanced Python libraries for image and data analysis, and a responsive front-end for data visualization.

✨ Key Features
Automated Plot Detection: Utilizes advanced image processing techniques to accurately identify and segment individual farmland plots from a single image.

Quantitative Land Analysis: Calculates and reports crucial metrics for each plot, including:

Total Area and Perimeter

Circularity and Solidity

Fallow vs. Non-Fallow Status

Actionable Insights: Provides a comprehensive summary of land utilization, including the fragmentation index and an irrigation advisory.

Data-Driven Visualizations: Generates a suite of informative plots using matplotlib to visualize key data points:

Histogram of Plot Areas

Box Plot of Plot Circularity

Pie Chart of Fallow vs. Non-Fallow Plots

Bar Chart of Irrigation Advisory Distribution

Advanced Clustering: Employs K-Means clustering and PCA to group plots based on their characteristics, revealing hidden patterns in the farmland.

Downloadable Reports: Allows users to download a detailed Excel spreadsheet containing the raw data and analysis results for every detected plot.

⚙️ Technologies Used
Backend & Data Analysis:

Python 3.x

Flask: Web framework for handling requests and serving the application.

OpenCV (cv2): Core library for image pre-processing and contour detection.

scikit-image (skimage): Used for region property analysis (regionprops).

NumPy: Essential for numerical operations on image data.

Pandas: Data manipulation and analysis, used to generate summary tables and Excel reports.

Matplotlib: Generates high-quality data visualizations and plots.

scikit-learn (sklearn): Powers the K-Means clustering and Principal Component Analysis (PCA).

Frontend:

HTML5

Bootstrap 5: Provides a responsive and clean user interface.

🚀 How It Works
Image Upload: A user uploads a satellite image of farmland via the web interface.

Preprocessing: The image is resized, converted to grayscale, blurred to reduce noise, and then processed with Otsu's thresholding and Canny edge detection.

Contour & Feature Extraction: The application finds all contours in the processed image, identifying potential plots. regionprops is used to calculate area, perimeter, and other features for each detected contour.

Data Analysis: The extracted features are compiled into a pandas DataFrame. K-Means clustering is applied to the data, and a PCA plot is generated to visualize the clusters.

Visualization & Output: matplotlib is used to create and save the various statistical plots. The results, including the original image with contours and plot numbers overlaid, are rendered on the index.html page along with an Excel download link.

🛠️ Installation
Prerequisites
Python 3.x

pip (Python package installer)

Clone the Repository:

git clone https://www.github.com/your-username/DetectFarm.git
cd DetectFarm

Create a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install Dependencies:

pip install Flask numpy opencv-python scikit-image pandas matplotlib scikit-learn

Note: A requirements.txt file is not included in the provided code, but this command will install all necessary packages.

▶️ Usage
Run the Flask Application:

python app.py

Open in Browser:
Open your web browser and navigate to http://127.0.0.1:5000.

Upload an Image:
Use the "Choose File" button to upload an image of farmland and click "Upload". The application will process the image and display the results.
