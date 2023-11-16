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
plt.plot(range(1, len(explained_variance_ratio_cumsum) + 1), explained_variance_ratio_cumsum, marker='o', color='green')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance ratio')
plt.grid(True)
plt.show()

pca_final = PCA(n_components=n_components_final)
X_pca_final = pca_final.fit_transform(X)  # 是否要特征缩放？

print("保留95%解释方差所要下降的维度：")
print(n_components_final)

# Task2

X_train, X_test, y_train, y_test = train_test_split(X_pca_final, y, test_size=0.2, random_state=42)  # 划分训练集 测试集

# SVM

svm = SVC(C=0.1, kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))

# DT
decisionTree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=5,
                                      random_state=42)  # change the parameter
decisionTree.fit(X_train, y_train)
y_pred_dt = decisionTree.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# RF
randomTree = RandomForestClassifier(max_depth=10, min_samples_split=10, n_estimators=1000, random_state=42)
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


# 绘制热力图
def plot_confusion_matrix(y_true, y_pred, title):
    matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Reds', square=True)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)


plt.figure(figsize=(10, 10))
plt.subplot(221)
plot_confusion_matrix(y_test, y_pred_svm, "SVM")
plt.subplot(222)
plot_confusion_matrix(y_test, y_pred_dt, "Decision Tree")
plt.subplot(223)
plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")
plt.subplot(224)
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.99, wspace=0.3, hspace=0.5)

plt.show()



# 使用5折交叉验证
cv = 5

# 计算混淆矩阵
svm_y_pred = cross_val_predict(svm, X_pca_final, y, cv=cv)
svm_cm = confusion_matrix(y, svm_y_pred)

dt_y_pred = cross_val_predict(decisionTree, X_pca_final, y, cv=cv)
dt_cm = confusion_matrix(y, dt_y_pred)

rf_y_pred = cross_val_predict(randomTree, X_pca_final, y, cv=cv)
rf_cm = confusion_matrix(y, rf_y_pred)

lr_y_pred = cross_val_predict(logistic_regression, X_pca_final, y, cv=cv)
lr_cm = confusion_matrix(y, lr_y_pred)

# 绘制混淆矩阵热力图
fig, ax = plt.subplots(2, 2, figsize=(12, 10))
sns.heatmap(svm_cm, annot=True, fmt='d', ax=ax[0, 0], cmap='Blues')
ax[0, 0].set_title('SVM Confusion Matrix')

sns.heatmap(dt_cm, annot=True, fmt='d', ax=ax[0, 1], cmap='Blues')
ax[0, 1].set_title('Decision Tree Confusion Matrix')

sns.heatmap(rf_cm, annot=True, fmt='d', ax=ax[1, 0], cmap='Blues')
ax[1, 0].set_title('Random Forest Confusion Matrix')

sns.heatmap(lr_cm, annot=True, fmt='d', ax=ax[1, 1], cmap='Blues')
ax[1, 1].set_title('Logistic Regression Confusion Matrix')

plt.tight_layout()
plt.show()
