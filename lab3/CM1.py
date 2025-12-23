import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import os

output_dir = 'cm1'

data, meta = arff.loadarff('CM1.arff')
df = pd.DataFrame(data)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(lambda x: x.decode('utf-8'))

# 提取特征 (X) 和类别标签 (y)
X = df.iloc[:, :-1]
y_names = df['Defective']

le = LabelEncoder()
y = le.fit_transform(y_names)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)


dt_full = DecisionTreeClassifier(random_state=42)
bag_full = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=50, random_state=42)
dt_full.fit(X_train, y_train)
bag_full.fit(X_train, y_train)
y_pred_dt_full = dt_full.predict(X_test)
y_pred_bag_full = bag_full.predict(X_test)

# 计算评估指标
def evaluate_binary(y_true, y_pred, y_prob):
    P = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    R = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    F1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    return P, R, F1, roc_auc, fpr, tpr

P_dt_full, R_dt_full, F1_dt_full, AUC_dt_full, fpr_dt_full, tpr_dt_full = evaluate_binary(y_test, y_pred_dt_full, dt_full.predict_proba(X_test))
P_bag_full, R_bag_full, F1_bag_full, AUC_bag_full, fpr_bag_full, tpr_bag_full = evaluate_binary(y_test, y_pred_bag_full, bag_full.predict_proba(X_test))

results = {
    'DT_Full': {'P': P_dt_full, 'R': R_dt_full, 'F1': F1_dt_full, 'AUC': AUC_dt_full},
    'Bagging_Full': {'P': P_bag_full, 'R': R_bag_full, 'F1': F1_bag_full, 'AUC': AUC_bag_full}
}

#特征选择：选择 K=10 个最佳特征
k_best = 10
selector = SelectKBest(score_func=f_classif, k=k_best)
X_train_kbest = selector.fit_transform(X_train, y_train)
X_test_kbest = selector.transform(X_test)
dt_kbest = DecisionTreeClassifier(random_state=42)
bag_kbest = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=50, random_state=42)

dt_kbest.fit(X_train_kbest, y_train)
bag_kbest.fit(X_train_kbest, y_train)


y_pred_dt_kbest = dt_kbest.predict(X_test_kbest)
y_pred_bag_kbest = bag_kbest.predict(X_test_kbest)

P_dt_kbest, R_dt_kbest, F1_dt_kbest, AUC_dt_kbest, fpr_dt_kbest, tpr_dt_kbest = evaluate_binary(y_test, y_pred_dt_kbest, dt_kbest.predict_proba(X_test_kbest))
P_bag_kbest, R_bag_kbest, F1_bag_kbest, AUC_bag_kbest, fpr_bag_kbest, tpr_bag_kbest = evaluate_binary(y_test, y_pred_bag_kbest, bag_kbest.predict_proba(X_test_kbest))

results.update({
    'DT_KBest': {'P': P_dt_kbest, 'R': R_dt_kbest, 'F1': F1_dt_kbest, 'AUC': AUC_dt_kbest},
    'Bagging_KBest': {'P': P_bag_kbest, 'R': R_bag_kbest, 'F1': F1_bag_kbest, 'AUC': AUC_bag_kbest}
})



df_results = pd.DataFrame.from_dict(results, orient='index')
df_results.columns = ['Precision (P)', 'Recall (R)', 'F1-Measure', 'AUC']

print("\n CM1 分类性能指标对比表")
print(df_results.round(4))

csv_path = os.path.join(output_dir, 'cm1_metrics_comparison.csv')
df_results.round(4).to_csv(csv_path)
print(f"\n 性能指标表格已保存至 '{csv_path}'")


# --- ROC 曲线对比图 ---
plt.figure(figsize=(10, 8))

plt.plot(fpr_dt_full, tpr_dt_full, label=f'DT Full (AUC = {AUC_dt_full:.3f})')
plt.plot(fpr_bag_full, tpr_bag_full, label=f'Bagging Full (AUC = {AUC_bag_full:.3f})')
plt.plot(fpr_dt_kbest, tpr_dt_kbest, label=f'DT KBest (AUC = {AUC_dt_kbest:.3f})')
plt.plot(fpr_bag_kbest, tpr_bag_kbest, label=f'Bagging KBest (AUC = {AUC_bag_kbest:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.5)') # 随机猜测模型
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('CM1 ROC Curve Comparison (Defective: Y vs N)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

# 保存图表
roc_fig_path = os.path.join(output_dir, 'cm1_roc_curve_comparison.png')
plt.savefig(roc_fig_path, dpi=300)
print(f" ROC曲线对比图已保存至 '{roc_fig_path}'")

plt.show()

