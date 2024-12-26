# Customer_segmentation

This Streamlit application allows users to upload a dataset containing customer information, visualize various data distributions, and perform clustering analysis using K-Means. The app is designed for exploratory data analysis (EDA) and customer segmentation.

## Features

- Upload a CSV file containing mall customer data.
- Automatically drops the `CustomerID` column for better visualization and analysis.
- Visualize data distributions with histograms, count plots, and violin plots.
- Group customers by age, spending score, and income to identify patterns.
- Perform K-Means clustering with options for 2D and 3D visualizations.
- Evaluate the optimal number of clusters using the Elbow Method.
- Visualize clustering results in 2D and 3D.

## Requirements

To run this application, ensure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `streamlit`

## Activate Python environment :
```bash
python -m venv venv

You can install them using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit

```bash
pip install -r requirements.txt

## Run 
```bash
streamlit run main.py
