import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score




data = pd.read_csv("Data.csv", sep=',', header=0)
# 提取数据成为矩阵

# 数据预处理：异常值2
data = data[data['Label'] != 2] # 删除label为2的
dataValues = data.iloc[:, 1:-1] # 这一行代码选择了除了最后一列以外的所有列作为数据。
labels = data.iloc[:, -1] # 表示选择所有行的最后一列作为label
print(dataValues)
X = dataValues
y = labels
# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 降维 - PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 计算累计解释方差比
n_components1 = 0
explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
for idx, cum_var in enumerate(explained_variance_ratio_cumsum):
    if cum_var >= 0.95: # 保留95%的解释方差
        n_components1 = idx + 1
        break
print(pca.explained_variance_ratio_)

# 绘制解释方差曲线图
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.plot(range(1, len(explained_variance_ratio_cumsum) + 1), explained_variance_ratio_cumsum, marker='o')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance ratio')
plt.grid(True)
plt.show()

pca_final = PCA(n_components=n_components1)
X_pca_final = pca_final.fit_transform(X_scaled)


print(n_components1)

X_train, X_test, y_train, y_test = train_test_split(X_pca_final, y, test_size= 0.3, random_state=42)

svm_classifier = SVC(C=1)
svm_classifier.fit(X_train, y_train)
y_pred_svm = svm_classifier.predict(X_test)

print("SVM分类器性能：")
print(classification_report(y_test, y_pred_svm))

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

print("决策树分类器性能：")
print(classification_report(y_test, y_pred_dt))

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 对测试集进行预测
y_pred_gnb = gnb.predict(X_test)

# 评估高斯朴素贝叶斯分类器的性能
print("高斯贝叶斯：")
print("Classification Report:\n", classification_report(y_test, y_pred_gnb))


# 获取预测结果的混淆矩阵
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_gnb = confusion_matrix(y_test, y_pred_gnb)

# 绘制热力图
plt.figure(figsize=(12,6))
plt.subplot(131)
sns.heatmap(cm_svm, annot=True, cmap="YlGnBu")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.subplot(132)
sns.heatmap(cm_dt, annot=True, cmap="YlGnBu")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.subplot(133)
sns.heatmap(cm_gnb, annot=True, cmap="YlGnBu")
plt.title("Gaussian Naive Bayes Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.show()

# 计算SVM的ROC曲线和AUC值
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_pred_svm)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)

# 计算决策树的ROC曲线和AUC值
fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)

# 计算高斯贝叶斯的ROC曲线和AUC值
fpr_gnb, tpr_gnb, thresholds_gnb = roc_curve(y_test, y_pred_gnb)
roc_auc_gnb = roc_auc_score(y_test, y_pred_gnb)

# 绘制ROC曲线
plt.figure(figsize=(8,6))
plt.plot(fpr_svm, tpr_svm, color='blue', label='SVM (AUC = %0.2f)' % roc_auc_svm)
plt.plot(fpr_dt, tpr_dt, color='green', label='Decision Tree (AUC = %0.2f)' % roc_auc_dt)
plt.plot(fpr_gnb, tpr_gnb, color='red', label='Gaussian Naive Bayes (AUC = %0.2f)' % roc_auc_gnb)

plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()