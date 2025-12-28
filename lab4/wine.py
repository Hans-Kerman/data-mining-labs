import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, f1_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

def align_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """使用匈牙利算法将聚类标签映射至真实标签"""
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    mapping = {col: row + 1 for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in y_pred])

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, X: np.ndarray, name: str) -> None:
    # 标签对齐 [cite: 69]
    y_pred_aligned = align_labels(y_true, y_pred)
    
    # 计算各项指标 [cite: 67, 71, 73, 80]
    mi_f1 = f1_score(y_true, y_pred_aligned, average='micro')
    ma_f1 = f1_score(y_true, y_pred_aligned, average='macro')
    sil_score = silhouette_score(X, y_pred)
    
    print(f"--- {name} ---")
    print(f"Micro-F1: {mi_f1:.4f}")
    print(f"Macro-F1: {ma_f1:.4f}")
    print(f"Silhouette: {sil_score:.4f}\n")

def main() -> None:
    # 加载数据，第一列为类别标识 (1-3) 
    data = pd.read_csv('./data/wine-data.txt', header=None)
    y_true = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values

    # 特征标准化 
    X_scaled = StandardScaler().fit_transform(X)

    # 1. K-Means 聚类 [cite: 11]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X_scaled)
    evaluate(y_true, y_kmeans, X_scaled, "K-Means")

    # 2. 凝聚层次聚类 [cite: 40, 42]
    hac = AgglomerativeClustering(n_clusters=3, linkage='ward')
    y_hac = hac.fit_predict(X_scaled)
    evaluate(y_true, y_hac, X_scaled, "HAC (Hierarchical)")

if __name__ == "__main__":
    main()