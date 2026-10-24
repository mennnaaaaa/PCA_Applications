# Principal Component Analysis (PCA) Applications

A practical machine learning notebook demonstrating **Principal Component Analysis (PCA)** for dimensionality reduction and visualization using the Breast Cancer Wisconsin dataset from Scikit-learn.

This repository focuses on understanding how PCA transforms a dataset with many numerical features into a smaller set of principal components while preserving as much of the data's variance as possible.

## Project Overview

The project applies PCA to the Breast Cancer Wisconsin dataset. The notebook:

1. Loads the dataset from Scikit-learn.
2. Converts the numerical feature data into a Pandas DataFrame.
3. Standardizes the features using `StandardScaler`.
4. Applies PCA to reduce the feature space to two principal components.
5. Visualizes the transformed samples in a two-dimensional scatter plot.
6. Examines the principal component loadings with a heatmap.

The dataset's target labels are used to color the scatter plot for visual comparison. PCA itself is an unsupervised dimensionality-reduction technique and does not use those labels when fitting the components.

## Repository Structure

```text
PCA_Applications/
└── PCA_PYTHON.ipynb
```

## File Description

| File | Purpose |
|---|---|
| `PCA_PYTHON.ipynb` | Complete PCA workflow using the Breast Cancer Wisconsin dataset, including data inspection, feature standardization, dimensionality reduction, two-dimensional visualization, and component-loading analysis. |

## What Is PCA?

**Principal Component Analysis (PCA)** is a statistical technique commonly used in machine learning to reduce the number of dimensions in a dataset.

A dataset may contain many features, some of which are correlated or contribute overlapping information. PCA creates new variables called **principal components**. Each component is a linear combination of the original features.

The components are ordered by the amount of variance they explain:

- **First principal component (PC1):** captures the greatest possible variance in a single direction.
- **Second principal component (PC2):** captures the greatest remaining variance while being orthogonal to PC1.
- **Additional components:** capture progressively less of the remaining variance.

By keeping only the first few components, PCA can represent a high-dimensional dataset in fewer dimensions, with some information loss.

## Why Do We Use PCA in Machine Learning?

PCA is useful for several reasons:

- **Dimensionality reduction:** reduces the number of input dimensions while retaining as much variance as possible.
- **Visualization:** makes datasets with many features easier to explore in two or three dimensions.
- **Redundancy reduction:** combines correlated features into a smaller set of uncorrelated components.
- **Computational efficiency:** can reduce the cost of downstream processing when fewer dimensions are sufficient.
- **Noise reduction:** removing low-variance components can sometimes reduce noise, although this depends on the dataset and should be validated.

PCA is not a classification algorithm. It transforms the feature representation. A separate model, such as logistic regression or an SVM, can be trained using the resulting components if a prediction task is required.

## Why Standardize the Data Before PCA?

The notebook standardizes the numerical features using Scikit-learn's `StandardScaler` before fitting PCA.

This matters because PCA is influenced by feature variance. If features are measured on very different scales, features with larger numerical ranges can dominate the principal components.

The notebook performs the following steps:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)
```

The transformed features have approximately zero mean and unit variance, making their contributions more comparable.

## Implementation in This Project

### 1. Load the Dataset

The notebook loads the Breast Cancer Wisconsin dataset directly from Scikit-learn:

```python
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
```

It inspects the dataset's keys and description, then creates a DataFrame containing the numerical measurements:

```python
df = pd.DataFrame(
    cancer["data"],
    columns=cancer["feature_names"]
)
```

The dataset contains multiple numerical features describing characteristics of cell nuclei. The target labels identify the dataset's two diagnostic classes.

### 2. Standardize the Features

The feature matrix is standardized with `StandardScaler` before PCA. The target labels are not included in the feature scaling or PCA fitting.

### 3. Fit PCA with Two Components

The notebook imports and configures PCA as follows:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(scaled_data)
```

Setting `n_components=2` instructs PCA to represent each sample using two principal components.

### 4. Transform the Dataset

The fitted PCA model transforms the standardized data:

```python
x_pca = pca.transform(scaled_data)
```

The original feature matrix has many columns, while the transformed representation has two columns: PC1 and PC2. This creates a compact representation suitable for a two-dimensional plot.

### 5. Visualize the Principal Components

The notebook plots the transformed samples:

```python
plt.scatter(
    x_pca[:, 0],
    x_pca[:, 1],
    c=cancer["target"]
)
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
```

Each point represents one sample. Its position is determined by its scores on PC1 and PC2, and its color represents the original target label.

This visualization helps explore whether the two classes appear separated in the reduced feature space. It does not by itself establish a model's predictive accuracy.

### 6. Inspect Principal Component Loadings

The notebook examines the component weights:

```python
pca.components_
```

It then places these weights into a DataFrame using the original feature names and visualizes them with a heatmap:

```python
df_comp = pd.DataFrame(
    pca.components_,
    columns=cancer["feature_names"]
)

sns.heatmap(df_comp, cmap="plasma")
```

The component weights, often called **loadings**, show how the original features contribute to each principal component. Their signs and magnitudes help interpret the directions identified by PCA.

## Machine Learning Workflow

```text
Load Breast Cancer Dataset
          |
          v
Inspect Dataset and Feature Names
          |
          v
Create Feature DataFrame
          |
          v
Standardize Numerical Features
          |
          v
Fit PCA (2 Components)
          |
          v
Transform Features into PC1 and PC2
          |
          v
Visualize Samples by Target Label
          |
          v
Analyze Component Loadings
```

## Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Installation and Usage

Clone the repository:

```bash
git clone https://github.com/mennnaaaaa/PCA_Applications.git
cd PCA_Applications
```

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `PCA_PYTHON.ipynb` and run the cells sequentially.

## Key Learning Outcomes

This project demonstrates how to:

- Explain the purpose of PCA in machine learning.
- Prepare numerical data for dimensionality reduction.
- Standardize features before applying PCA.
- Fit PCA with a chosen number of components.
- Transform a high-dimensional dataset into two dimensions.
- Visualize transformed data using principal component scores.
- Inspect component loadings to understand feature contributions.
- Recognize that PCA is a feature transformation technique rather than a classifier.

## Limitations and Considerations

- PCA is a linear transformation and may not preserve nonlinear structure.
- Dimensionality reduction can discard information, so the explained variance should be examined when deciding how many components to retain.
- The principal components can be harder to interpret than the original features.
- The two-dimensional plot is useful for exploration, but visual separation does not replace quantitative evaluation when building a predictive model.
- If PCA is used before training a supervised model, fit the scaler and PCA on the training data only, then apply the fitted transformations to validation and test data to avoid data leakage.

## Project Background

This repository contains a practical PCA implementation for studying dimensionality reduction and data visualization in machine learning. The notebook uses Scikit-learn's Breast Cancer Wisconsin dataset to demonstrate the full transformation workflow and interpretation of principal components.

## Author

**Menna Hany Abdelaziz**

GitHub: [mennnaaaaa](https://github.com/mennnaaaaa)
