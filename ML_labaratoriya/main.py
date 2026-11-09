import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from umap import UMAP

# Dataset va hisob-kitoblar
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
X_scaled = StandardScaler().fit_transform(X)

X_pca = PCA(n_components=2).fit_transform(X_scaled)
X_lda = LDA(n_components=2).fit_transform(X_scaled, y)
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
X_umap = UMAP(n_components=2, random_state=42).fit_transform(X_scaled)

# Grafiklarni chiqarish
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
axes[0, 0].set_title('PCA')
axes[0, 1].scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis')
axes[0, 1].set_title('LDA')
axes[1, 0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
axes[1, 0].set_title('t-SNE')
axes[1, 1].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='viridis')
axes[1, 1].set_title('UMAP')
plt.show()
