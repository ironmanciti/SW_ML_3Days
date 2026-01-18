# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fraud Payment Logistic Regression (모범 답안)
#
# 이 파일은 Spam Filtering 실습의 Fraud Payment 문제에 대한 모범 답안입니다.
# 수강생들이 자신의 문제 풀이와 비교하여 학습할 수 있도록 작성되었습니다.
#
# 결제 사기 탐지 문제를 로지스틱 회귀를 사용하여 해결합니다.
# - accountAgeDays : 계정이 생성된 기간 (일)
# - numItems : 구매한 항목 수
# - localTime : 결제가 이루어진 시기 (부동 숫자로 변환 됨)
# - paymentMethod : 결제 방법 (페이팔, 상점 신용 카드 또는 신용 카드)
# - paymentMethodAgeDays : 결제가 완료된 기간 (일)
# - label 0 - 정상, 1 - fraud

# %% [markdown]
# 필요한 라이브러리를 import 합니다. 로지스틱 회귀와 평가 지표를 사용합니다.

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# %% [markdown]
# 결제 사기 탐지 데이터셋을 로드합니다. 이 데이터로 정상 결제와 사기 결제를 구분하는 모델을 학습합니다.

df = pd.read_csv("security_data/payment_fraud.csv")
df.sample(5)

# %% [markdown]
# 라벨의 분포를 확인합니다. 정상(0)과 사기(1)의 비율을 확인할 수 있습니다. 데이터 불균형이 있을 수 있습니다.

# %%
df.label.value_counts()

# %% [markdown]
# 데이터 불균형 문제를 해결하기 위해 사기(1) 샘플을 모두 선택하고, 정상(0) 샘플은 일부만 샘플링합니다.

# %%
df1 = df[df.label == 1]
df1.shape

# %% [markdown]
# 정상 결제 데이터에서 1000개를 무작위로 샘플링합니다. 이렇게 하면 클래스 불균형을 완화할 수 있습니다.

# %%
df0 = df[df.label == 0].sample(1000)
df0.shape

# %% [markdown]
# 샘플링된 정상 데이터와 전체 사기 데이터를 합쳐서 새로운 데이터셋을 만듭니다.

# %%
df_new = pd.concat([df0, df1])
df_new.shape

# %% [markdown]
# 결제 방법을 원-핫 인코딩으로 변환합니다. 각 결제 방법이 별도의 이진 특성으로 변환됩니다.

# %%
df_new = pd.get_dummies(df_new, columns=['paymentMethod'])

# %% [markdown]
# 원-핫 인코딩이 적용된 데이터를 확인합니다. paymentMethod 컬럼이 여러 개의 이진 컬럼으로 분리되었습니다.

# %%
df_new.sample(5)

# %% [markdown]
# ### Train / Test split

# %% [markdown]
# 특성과 라벨을 분리합니다. pop() 메서드를 사용하여 label 컬럼을 제거하면서 동시에 y로 할당합니다.

# %%
y = df_new.pop('label')
X = df_new.values

# %% [markdown]
# 데이터를 훈련용과 테스트용으로 분할합니다. test_size=0.2로 80%는 훈련, 20%는 테스트 데이터로 나뉩니다.

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# ### Model 생성

# %% [markdown]
# 로지스틱 회귀 모델을 생성하고 훈련 데이터로 학습시킵니다. 결제 정보를 기반으로 사기를 탐지합니다.

# %%
lr_classifier = LogisticRegression(solver='lbfgs', random_state=0)
lr_classifier.fit(X_train, y_train)

# %% [markdown]
# ### predict
#
# - predict() - 예측된 class 를 threshold 0.5 기준으로 반환
# - predict_proba() - class 별 probability 를 반환

# %% [markdown]
# 학습된 모델로 테스트 데이터에 대한 예측을 수행합니다. 각 결제가 정상인지 사기인지 분류합니다.

# %%
y_pred = lr_classifier.predict(X_test)

print(y_pred)
print()
print("Test set 의 true counts = ", sum(y_test))
print("모델이 예측한 predicted true counts = ", sum(y_pred))
print("accuracy = {:.2f}".format(sum(y_pred == y_test) / len(y_test)))

# %% [markdown]
# ## confusion matrix 를 이용한 model 평가

# %% [markdown]
# 평가 지표를 계산하기 위한 함수들을 import 합니다. confusion matrix와 다양한 성능 지표를 사용합니다.

# %%
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import  accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict
import seaborn as sns

# %% [markdown]
# 혼동 행렬을 계산하고 시각화합니다. 실제 라벨과 예측 라벨의 일치 여부를 확인할 수 있습니다. 사기 탐지에서는 False Negative(사기를 놓치는 경우)가 특히 중요합니다.

# %%
cm  = confusion_matrix(y_test, y_pred, labels=[1, 0])

print("Confusion Matrix:")
print(cm)
print("\n[행: 실제값, 열: 예측값]")
print("  Fraud(1)  Normal(0)")
print(f"Fraud(1)   {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"Normal(0)  {cm[1,0]:4d}  {cm[1,1]:4d}")

plt.figure(figsize=(8, 6))

# 더 명확한 시각화
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                 xticklabels=['Fraud (1)', 'Normal (0)'],
                 yticklabels=['Fraud (1)', 'Normal (0)'],
                 cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix - Fraud Payment Detection', fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# 각 셀의 의미 설명
print("\nConfusion Matrix 해석:")
print(f"True Positive (TP): {cm[0,0]} - 사기를 사기로 정확히 예측")
print(f"False Positive (FP): {cm[0,1]} - 정상을 사기로 잘못 예측")
print(f"False Negative (FN): {cm[1,0]} - 사기를 정상으로 잘못 예측 (위험!)")
print(f"True Negative (TN): {cm[1,1]} - 정상을 정상으로 정확히 예측")

# %% [markdown]
# 정확도, 정밀도, 재현율, F1 점수를 계산하여 모델의 성능을 종합적으로 평가합니다.

# %%
print("Accuracy: {:.4f}".format(accuracy_score(y_test, y_pred)))
print("Precision: {:.4f}".format(precision_score(y_test, y_pred, labels=[1, 0])))
print("Recall: {:.4f}".format(recall_score(y_test, y_pred, labels=[1, 0])))
print("F1 Score: {:.4f}".format(f1_score(y_test, y_pred)))
