import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import streamlit as st



st.title("Mall Customer Segmentation Analysis")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.drop(['CustomerID'], axis=1, inplace=True)

    st.subheader("Data Overview")
    st.write(df.head())

    st.subheader("Data Distribution")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for i, col in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
        sns.histplot(df[col], bins=20, ax=axes[i])
        axes[i].set_title(f'Distplot of {col}')
    st.pyplot(fig)

    st.subheader("Gender Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(y='Gender', data=df, hue='Gender', ax=ax)
    st.pyplot(fig)

    st.subheader("Violin Plots")
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    for i, col in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
        sns.violinplot(x=col, y='Gender', data=df, hue='Gender', ax=axes[i])
        axes[i].set_ylabel('Gender' if i == 0 else '')
        axes[i].set_title('Violin Plot')
    st.pyplot(fig)

    st.subheader("Age Distribution")
    age_ranges = [
        ('18-25', 18, 25),
        ('26-35', 26, 35),
        ('36-45', 36, 45),
        ('46-55', 46, 55),
        ('55+', 56, 100)
    ]
    age_counts = [len(df[(df['Age'] >= low) & (df['Age'] <= high)]) for _, low, high in age_ranges]
    
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(x=[r[0] for r in age_ranges], y=age_counts, palette="mako", ax=ax)
    ax.set_title("Number of Customers by Age Group")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.subheader("Annual Income vs Spending Score")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Spending Score Distribution")
    score_ranges = [
        ('1-20', 1, 20),
        ('21-40', 21, 40),
        ('41-60', 41, 60),
        ('61-80', 61, 80),
        ('81-100', 81, 100)
    ]
    score_counts = [len(df[(df['Spending Score (1-100)'] >= low) & (df['Spending Score (1-100)'] <= high)]) for _, low, high in score_ranges]
    
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(x=[r[0] for r in score_ranges], y=score_counts, palette="rocket", ax=ax)
    ax.set_title("Spending Scores Distribution")
    ax.set_xlabel("Score Range")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.subheader("Annual Income Distribution")
    income_ranges = [
        ('$0-30,000', 0, 30),
        ('$31,000-60,000', 31, 60),
        ('$61,000-90,000', 61, 90),
        ('$91,000-120,000', 91, 120),
        ('$120,001-150,000', 121, 150)
    ]
    income_counts = [len(df[(df['Annual Income (k$)'] >= low) & (df['Annual Income (k$)'] <= high)]) for _, low, high in income_ranges]
    
    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(x=[r[0] for r in income_ranges], y=income_counts, palette="Spectral", ax=ax)
    ax.set_title("Annual Income Distribution")
    ax.set_xlabel("Income Range")
    ax.set_ylabel("Number of Customers")
    st.pyplot(fig)

    st.subheader("K-Means Clustering")
    
    X1 = df.loc[:, ["Age", "Spending Score (1-100)"]].values
    X2 = df.loc[:, ["Annual Income (k$)", "Spending Score (1-100)"]].values
    X3 = df.iloc[:, 1:].values

    def plot_elbow(X, title):
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(1, 11), wcss, linewidth=2, color="red", marker="8")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("WCSS")
        ax.set_title(f"The Elbow Method for {title}")
        ax.grid(True)
        st.pyplot(fig)

    plot_elbow(X1, "Age vs Spending Score")
    plot_elbow(X2, "Annual Income vs Spending Score")
    plot_elbow(X3, "All Features")

    def plot_clusters(X, n_clusters, xlabel, ylabel):
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(X)

        fig, ax = plt.subplots(figsize=(12, 8))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow")
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="black", s=100, marker="x")
        ax.set_title(f"Clusters of Customers ({xlabel} vs {ylabel})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(scatter)
        st.pyplot(fig)

    st.subheader("2D Clustering Results")
    plot_clusters(X1, 4, "Age", "Spending Score (1-100)")
    plot_clusters(X2, 5, "Annual Income (k$)", "Spending Score (1-100)")

    st.subheader("3D Clustering Results")
    kmeans = KMeans(n_clusters=5)
    labels = kmeans.fit_predict(X3)
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(df.Age, df["Annual Income (k$)"], df["Spending Score (1-100)"], c=labels, cmap="rainbow")
    ax.set_xlabel("Age")
    ax.set_ylabel("Annual Income (k$)")
    ax.set_zlabel("Spending Score (1-100)")
    ax.view_init(30, 185)
    plt.colorbar(scatter)
    st.pyplot(fig)

else:
    st.write("Please upload a CSV file to begin the analysis.")