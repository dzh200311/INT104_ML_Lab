import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples
import matplotlib.cm as cm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from sklearn.metrics import precision_recall_curve, average_precision_score

data = pd.read_csv("Data.csv", sep=',', header=0)

#  数据预处理：异常值2
data = data[data['Label'] != 2]  # 删除label为2的
dataValue = data.iloc[:, 1:-1]  # 这一行代码选择了除了最后一列以外的所有列作为数据。 patient idx 删掉
labels = data.iloc[:, -1]  # 表示选择所有行的最后一列作为label
print(dataValue)
X = dataValue
y = labels

corr_matrix = dataValue[['F2', 'F5', 'F6']].corr()
print("Correlation matrix for features F2, F5, and F6:")
print(corr_matrix)

# 特征缩放？
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# PCA
# 不改变原始数据进行降维
# 高维的缺点：可视化，过拟合，复杂度，冗余数据
pca = PCA()
X_pca = pca.fit(X)
# 计算累计解释方差比
n_components_final = 0
explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
print("累计解释方差：")
print(explained_variance_ratio_cumsum)
for index, cum_var in enumerate(explained_variance_ratio_cumsum):
    if cum_var >= 0.95:  # 保留95%的解释方差
        n_components_final = index + 1  # 当前的解释方差累计和是否大于等于 0.95，如果是，则设置变量 n_components1 为当前索引
        break

print("解释方差：")
print(pca.explained_variance_ratio_)

# 绘制解释方差曲线图
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.plot(range(1, len(explained_variance_ratio_cumsum) + 1), explained_variance_ratio_cumsum, marker='o')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance ratio')
plt.grid(True)
plt.show()

pca_final = PCA(n_components=10)
X_pca_final = pca_final.fit_transform(X)  # 是否要特征缩放？

print("保留95%解释方差所要下降的维度：")
print(n_components_final)

# Task2

X_train, X_test, y_train, y_test = train_test_split(X_pca_final, y, test_size=0.2, random_state=42)  # 划分训练集 测试集

# SVM

svm = SVC(C=0.1, kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# DT
decisionTree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, min_samples_split=2,
                                      random_state=42)  # change the parameter
decisionTree.fit(X_train, y_train)
y_pred_dt = decisionTree.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# RF
randomTree = RandomForestClassifier(max_depth=10, min_samples_split=5, n_estimators=1000, random_state=42)
randomTree.fit(X_train, y_train)
y_pred_rf = randomTree.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# LR
logistic_regression = LogisticRegression(C=0.1, solver='liblinear', random_state=42)
logistic_regression.fit(X_train, y_train)

# 对测试集进行预测
y_pred_lr = logistic_regression.predict(X_test)
y_pred_train_lr = logistic_regression.predict(X_train)

# 评估逻辑回归分类器的性能
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report:\n", classification_report(y_test, y_pred_lr))
print("Logistic Regression Train Accuracy:", accuracy_score(y_train, y_pred_train_lr))
cv = 5  # 使用 5 折交叉验证


# # SVM
# svm_cv_scores = cross_val_score(svm, X_pca_final, y, cv=cv)
# print("SVM Cross Validation Scores:", svm_cv_scores)
# print("SVM Cross Validation Mean Accuracy:", svm_cv_scores.mean())
#
# # DT
# dt_cv_scores = cross_val_score(decisionTree, X_pca_final, y, cv=cv)
# print("Decision Tree Cross Validation Scores:", dt_cv_scores)
# print("Decision Tree Cross Validation Mean Accuracy:", dt_cv_scores.mean())
#
# # RF
# rf_cv_scores = cross_val_score(randomTree, X_pca_final, y, cv=cv)
# print("Random Forest Cross Validation Scores:", rf_cv_scores)
# print("Random Forest Cross Validation Mean Accuracy:", rf_cv_scores.mean())
#
# # LR
# lr_cv_scores = cross_val_score(logistic_regression, X_pca_final, y, cv=cv)
# print("Logistic Regression Cross Validation Scores:", lr_cv_scores)
# print("Logistic Regression Cross Validation Mean Accuracy:", lr_cv_scores.mean())


# 绘制热力图
def plot_confusion_matrix(y_true, y_pred, title):
    matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Reds', square=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)


plt.figure(figsize=(15, 5))
plt.subplot(221)
plot_confusion_matrix(y_test, y_pred_svm, "SVM")
plt.subplot(222)
plot_confusion_matrix(y_test, y_pred_dt, "Decision Tree")
plt.subplot(223)
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")
plt.subplot(224)
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")
plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.4, hspace=0.2)

plt.show()


# 绘制ROC曲线
def plot_roc_curve(y_true, y_score, label, color):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC = {roc_auc:.2f})")


svm_score = svm.decision_function(X_test)
dt_score = decisionTree.predict_proba(X_test)[:, 1]
rf_score = randomTree.predict_proba(X_test)[:, 1]
lr_score = logistic_regression.predict_proba(X_test)[:, 1]

plt.figure(figsize=(8, 6))
plot_roc_curve(y_test, svm_score, "SVM", "blue")
plot_roc_curve(y_test, dt_score, "Decision Tree", "green")
plot_roc_curve(y_test, rf_score, "Random Forest", "red")

plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.show()

# # 绘制分类器的评估指标比较柱状图
# def plot_bar_chart(metrics, labels, title):
#     classifiers = ["SVM", "Decision Tree", "Random Forest", "Logistic Regression"]
#     plt.figure(figsize=(8, 6))
#     sns.barplot(x=classifiers, y=metrics)
#
#     # 在柱子上方显示数据
#     for i, v in enumerate(metrics):
#         plt.text(i, v + 0.01, str(round(v, 3)), ha='center')
#
#     plt.xlabel("Classifier")
#     plt.ylabel(labels)
#     plt.title(title)
#     plt.show()


# 计算评估指标
recall_scores = [
    recall_score(y_test, y_pred_svm),
    recall_score(y_test, y_pred_dt),
    recall_score(y_test, y_pred_rf),
    recall_score(y_test, y_pred_lr),
]

precision_scores = [
    precision_score(y_test, y_pred_svm),
    precision_score(y_test, y_pred_dt),
    precision_score(y_test, y_pred_rf),
    precision_score(y_test, y_pred_lr),
]
#
# # 绘制召回率比较柱状图
# plot_bar_chart(recall_scores, "Recall", "Classifier Recall Comparison")
#
# # 绘制精确率比较柱状图
# plot_bar_chart(precision_scores, "Precision", "Classifier Precision Comparison")

# 召回率就是真正的正确的比真正正确的加假错误的（就是明明是5但是判断的4）
# 精度就是真正正确的比真正正确的加假正确的（明明是4但是判断的是5）
# fallout就是错误被分为正确的占所有负类的比率（假正类是负类，真负类是负类，假负类是正类，真正类是正类）
# f1 = 2 * (p+R)/p*R
# ROC曲线的虚线表示纯随机分类器的曲线


# 计算轮廓系数以确定最佳聚类数
silhouette_scores = []
cluster_range = range(2, 11)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca_final)
    silhouette_avg = silhouette_score(X_pca_final, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# 绘制轮廓系数图
plt.figure(figsize=(8, 6))
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs. Number of Clusters")
plt.grid(True)
plt.show()

X = X_pca_final
cluster_range = range(2, 11)

for n_clusters in cluster_range:
    # 创建一个子图
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(8, 6)

    # 轮廓系数范围为[-1, 1]，为了美观起见，我们将y轴范围设置为[-0.1, 1]
    ax1.set_xlim([-0.1, 1])

    # 插入一些空间，以便在每个簇之间有间隔
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)

    # 计算每个样本的轮廓系数
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # 为每个簇添加簇编号
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # 为下一个簇预留空间

    ax1.set_title(f"Silhouette plot for {n_clusters} clusters")
    ax1.set_xlabel("Silhouette Coefficient")
    ax1.set_ylabel("Cluster label")

    # 在图上标注平均轮廓系数
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # 清除y轴上的刻度
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

# 绘制惯性图
inertia_values = []
cluster_range = range(2, 11)

for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(cluster_range, inertia_values, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Inertia vs. Number of Clusters")
plt.grid(True)
plt.show()

# 找到拐点
inertia_diff = [inertia_values[i] - inertia_values[i + 1] for i in range(len(inertia_values) - 1)]

print("Inertia differences:", inertia_diff)

elbow_point = -1
for i in range(1, len(inertia_diff) - 1):
    if inertia_diff[i - 1] > inertia_diff[i] > inertia_diff[i + 1]:
        elbow_point = i + 2  # Add 2 because the cluster_range starts from 2
        break

print("Elbow point (best number of clusters):", elbow_point)

# 使用选择的聚类数运行K-means算法
optimal_cluster_num = 2
kmeans = KMeans(n_clusters=optimal_cluster_num, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca_final)
silhouette0 = silhouette_score(X_pca_final, cluster_labels)
print(f"K Silhouette Coefficient: {silhouette0}")

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_pca_final)

# # 自定义参数网格
# param_grid = {
#     'eps': np.arange(0.1, 1.0, 0.1),
#     'min_samples': range(2, 20)
# }
#
# best_score = -1
# best_params = None
#
# # 遍历参数组合
# for params in ParameterGrid(param_grid):
#     dbscan = DBSCAN(**params)
#     labels = dbscan.fit_predict(X_normalized)
#
#     if len(set(labels)) == 1 or -1 not in labels:
#         continue
#
#     score = silhouette_score(X_normalized, labels)
#
#     if score > best_score:
#         best_score = score
#         best_params = params
#
# # 输出最佳参数
# print("Best parameters: ", best_params)
# print("Best score: ", best_score)
# best_eps = 0
# best_min_samples = 0
# best_silhouette = -1
#
# eps_values = np.arange(0.1, 0.6, 0.01)
# min_samples_values = np.arange(30, 60, 1)
#
# for eps in eps_values:
#     for min_samples in min_samples_values:
#         dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#         cluster_labels = dbscan.fit_predict(X_normalized)
#
#         # 如果只生成一个簇（或者所有点都是噪声），跳过这个参数组合
#         if len(set(cluster_labels)) <= 2 or -1 not in cluster_labels:
#             continue
#
#         silhouette = silhouette_score(X_normalized, cluster_labels)
#
#         if silhouette > best_silhouette:
#             best_silhouette = silhouette
#             best_eps = eps
#             best_min_samples = min_samples
#
# print(f"Best DBSCAN parameters: eps={best_eps}, min_samples={best_min_samples}")
# print(f"Best Silhouette Coefficient: {best_silhouette}")

dbscan = DBSCAN(eps=0.38, min_samples=30)  # 0.2：-0.221684；0.1：-0.234268； 0.3：-0.11699444916535749
# 0.4：0.08824714049303418 0.5: 0.135870(100:0.06754)
cluster_labels1 = dbscan.fit_predict(X_normalized)
n_clusters = len(set(cluster_labels1)) - (1 if -1 in cluster_labels1 else 0)

silhouette = silhouette_score(X_normalized, cluster_labels1)
print(f"D Silhouette Coefficient: {silhouette}")

from sklearn.manifold import TSNE

# 使用t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca_final)

# 可视化K-means聚类结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for cluster_label in range(optimal_cluster_num):
    plt.scatter(X_tsne[cluster_labels == cluster_label, 0], X_tsne[cluster_labels == cluster_label, 1],
                label=f'Cluster {cluster_label}')
plt.title('K-means Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()

# 可视化DBSCAN聚类结果
plt.subplot(1, 2, 2)
for cluster_label in range(n_clusters):
    plt.scatter(X_tsne[cluster_labels1 == cluster_label, 0], X_tsne[cluster_labels1 == cluster_label, 1],
                label=f'Cluster {cluster_label}')
# 显示噪声点
plt.scatter(X_tsne[cluster_labels1 == -1, 0], X_tsne[cluster_labels1 == -1, 1], c='black', marker='x', label='Noise')
plt.title('DBSCAN Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()

plt.tight_layout()
plt.show()

noise_points = np.sum(cluster_labels1 == -1)

print(f"Number of noise points: {noise_points}")

# 创建t-SNE模型并进行降维
tsne = TSNE(n_components=2, perplexity=30, n_iter=10000, random_state=42)
X_tsne = tsne.fit_transform(X_pca_final)
# 创建散点图并显示降维结果
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, alpha=0.5)
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.title('t-SNE Visualization of X_pca_final')
plt.show()
