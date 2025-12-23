import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import pandas as pd
import os 

output_dir = 'iris'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv('iris.csv')

# 特征列名（去掉第一列编号）
feature_names = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
X = df[feature_names].values
# 将 Species 映射成数字标签
species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
y = df['Species'].map(species_map).values
# 获取类别名称
target_names = np.array(['setosa', 'versicolor', 'virginica'])


# 数据标准化（PCA和KNN对尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)


# 初始化模型
knn_full = KNeighborsClassifier(n_neighbors=3)
nb_full = GaussianNB()

# 训练模型
knn_full.fit(X_train, y_train)
nb_full.fit(X_train, y_train)

# 预测
y_pred_knn_full = knn_full.predict(X_test)
y_pred_nb_full = nb_full.predict(X_test)

# 计算评估指标
results_full = {
    'KNN_Full': {
        'P': precision_score(y_test, y_pred_knn_full, average='macro'),
        'R': recall_score(y_test, y_pred_knn_full, average='macro'),
        'F1': f1_score(y_test, y_pred_knn_full, average='macro')
    },
    'NB_Full': {
        'P': precision_score(y_test, y_pred_nb_full, average='macro'),
        'R': recall_score(y_test, y_pred_nb_full, average='macro'),
        'F1': f1_score(y_test, y_pred_nb_full, average='macro')
    }
}


# PCA 降维到 2 维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 划分降维后的训练集和测试集
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.3, random_state=42, stratify=y
)

# 初始化并训练模型
knn_pca = KNeighborsClassifier(n_neighbors=3)
nb_pca = GaussianNB()

knn_pca.fit(X_train_pca, y_train_pca)
nb_pca.fit(X_train_pca, y_train_pca)

# 预测
y_pred_knn_pca = knn_pca.predict(X_test_pca)
y_pred_nb_pca = nb_pca.predict(X_test_pca)

# 计算评估指标
results_pca = {
    'KNN_PCA': {
        'P': precision_score(y_test_pca, y_pred_knn_pca, average='macro'),
        'R': recall_score(y_test_pca, y_pred_knn_pca, average='macro'),
        'F1': f1_score(y_test_pca, y_pred_knn_pca, average='macro')
    },
    'NB_PCA': {
        'P': precision_score(y_test_pca, y_pred_nb_pca, average='macro'),
        'R': recall_score(y_test_pca, y_pred_nb_pca, average='macro'),
        'F1': f1_score(y_test_pca, y_pred_nb_pca, average='macro')
    }
}


all_results = {**results_full, **results_pca}
df_results = pd.DataFrame.from_dict(all_results, orient='index')
df_results.columns = ['Precision (P)', 'Recall (R)', 'F1-Measure']

print("### 分类性能指标对比表 (P, R, F1-Measure)")
print(df_results)

# 绘制条形图对比
df_results.plot(kind='bar', figsize=(12, 7), rot=0) 
plt.title('Performance Comparison (Full Features vs. PCA)')
plt.ylabel('Score')
plt.ylim(0.8, 1.05) # 设定Y轴范围，突出差异
plt.grid(axis='y', linestyle='--')
plt.tight_layout() 


plt.savefig(os.path.join(output_dir, 'p_r_f1_comparison.png'), dpi=300)
print(f"\n P/R/F1对比图已保存至 '{output_dir}/p_r_f1_comparison.png'")

plt.show()



y_prob_knn_full = knn_full.predict_proba(X_test)
y_prob_nb_full = nb_full.predict_proba(X_test)
y_prob_knn_pca = knn_pca.predict_proba(X_test_pca)
y_prob_nb_pca = nb_pca.predict_proba(X_test_pca)

# 存储模型和对应的概率
models = {
    'KNN_Full': (y_prob_knn_full, y_test),
    'NB_Full': (y_prob_nb_full, y_test),
    'KNN_PCA': (y_prob_knn_pca, y_test_pca),
    'NB_PCA': (y_prob_nb_pca, y_test_pca)
}

# 绘制 ROC 曲线和计算 AUC
plt.figure(figsize=(10, 8))

# 存储所有模型的宏平均AUC，用于最终的表格输出
auc_results = {}

for name, (y_prob, y_true) in models.items():
    # 对每个类别计算 OvR AUC
    auc_scores = []
    # 用于绘制宏平均ROC曲线,需要一个通用的fpr网格
    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = 0.0
    
    for i in range(len(target_names)):
        # 类别 i 的 True/False
        y_i_true = (y_true == i).astype(int)
        y_i_prob = y_prob[:, i]

        fpr, tpr, _ = roc_curve(y_i_true, y_i_prob)
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)
        # 为了绘制平均ROC，对tpr进行插值
        mean_tpr += np.interp(all_fpr, fpr, tpr)
        mean_tpr[0] = 0.0


    # 绘制宏平均 ROC 曲线（非单一类别曲线）
    mean_tpr /= len(target_names)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(auc_scores) # 使用之前计算的auc平均值
    auc_results[name] = mean_auc # 存储AUC结果

    plt.plot(all_fpr, mean_tpr, label=f'{name} (Macro-Avg AUC = {mean_auc:.3f})')
    
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.5)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve Comparison (One-vs-Rest Macro-Averaged)')
plt.legend(loc="lower right")
plt.grid(True)


plt.savefig(os.path.join(output_dir, 'roc_curve_comparison.png'), dpi=300)
print(f" ROC曲线对比图已保存至 '{output_dir}/roc_curve_comparison.png'")

plt.show()


df_results['AUC (Macro Avg)'] = pd.Series(auc_results)
print("\n最终全性能指标对比表 (P, R, F1, AUC)")
print(df_results)
