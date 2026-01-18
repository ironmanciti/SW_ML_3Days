# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fraud Payment Detection - 종합 분류 모델 비교 (모범 답안)
#
# 이 파일은 결제 사기 탐지 문제를 여러 분류 모델로 해결하고 성능을 비교하는 종합 문제입니다.
# 수강생들이 자신의 문제 풀이와 비교하여 학습할 수 있도록 작성되었습니다.
#
# **학습 목표:**
# - 여러 분류 모델(Logistic Regression, KNN, Decision Tree, Random Forest)을 동일한 데이터에 적용
# - 각 모델의 성능을 비교하고 분석
# - 사기 탐지 문제에 가장 적합한 모델 선택
#
# **데이터 설명:**
# - accountAgeDays : 계정이 생성된 기간 (일)
# - numItems : 구매한 항목 수
# - localTime : 결제가 이루어진 시기 (부동 숫자로 변환 됨)
# - paymentMethod : 결제 방법 (페이팔, 상점 신용 카드 또는 신용 카드)
# - paymentMethodAgeDays : 결제가 완료된 기간 (일)
# - label 0 - 정상, 1 - fraud

# %% [markdown]
# ## 1. 데이터 로드 및 전처리

# %% [markdown]
# 필요한 라이브러리를 import 합니다. 여러 분류 모델과 평가 지표를 사용합니다.

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, 
                          precision_score, recall_score, roc_auc_score)
import seaborn as sns

# %% [markdown]
# 결제 사기 탐지 데이터셋을 로드합니다. 이 데이터로 정상 결제와 사기 결제를 구분하는 모델을 학습합니다.

# %%
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
df0 = df[df.label == 0].sample(1000, random_state=0)
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
# 특성과 라벨을 분리합니다. pop() 메서드를 사용하여 label 컬럼을 제거하면서 동시에 y로 할당합니다.

# %%
y = df_new.pop('label')
X = df_new.values

# %% [markdown]
# 데이터를 훈련용과 테스트용으로 분할합니다. test_size=0.2로 80%는 훈련, 20%는 테스트 데이터로 나뉩니다.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %% [markdown]
# 특성 스케일링을 수행합니다. StandardScaler를 사용하여 모든 특성을 평균 0, 표준편차 1로 정규화합니다.
# 일부 모델(KNN, Logistic Regression)은 스케일링이 중요합니다.

# %%
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# %% [markdown]
# ## 2. 여러 분류 모델 적용 및 평가

# %% [markdown]
# ### 2.1 Logistic Regression

# %% [markdown]
# 로지스틱 회귀 모델을 생성하고 훈련 데이터로 학습시킵니다. 선형 분류 모델로 빠르고 해석이 용이합니다.

# %%
lr_classifier = LogisticRegression(solver='liblinear', random_state=0, max_iter=1000)
lr_classifier.fit(X_train_scaled, y_train)

# %% [markdown]
# 테스트 데이터에 대한 예측을 수행합니다.

# %%
y_pred_lr = lr_classifier.predict(X_test_scaled)

print("Logistic Regression 예측 결과 (일부):", y_pred_lr[:10])
print(f"테스트 세트의 실제 사기 거래 수: {sum(y_test)}")
print(f"모델이 예측한 사기 거래 수: {sum(y_pred_lr)}")
print(f"정확도: {accuracy_score(y_test, y_pred_lr):.4f}")

# %% [markdown]
# ### 2.2 K-Nearest Neighbors (KNN)

# %% [markdown]
# KNN 분류기를 생성하고 학습시킵니다. KNN은 거리 기반 분류 모델로, 스케일링이 중요합니다.

# %%
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)

# %% [markdown]
# 테스트 데이터에 대한 예측을 수행합니다.

# %%
y_pred_knn = knn_classifier.predict(X_test_scaled)

print("KNN 예측 결과 (일부):", y_pred_knn[:10])
print(f"테스트 세트의 실제 사기 거래 수: {sum(y_test)}")
print(f"모델이 예측한 사기 거래 수: {sum(y_pred_knn)}")
print(f"정확도: {accuracy_score(y_test, y_pred_knn):.4f}")

# %% [markdown]
# ### 2.3 Decision Tree

# %% [markdown]
# Decision Tree 분류기를 생성하고 학습시킵니다. 트리 모델은 스케일링이 필요 없으며, 해석이 용이합니다.

# %%
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=0, criterion='entropy')
dt_classifier.fit(X_train, y_train)

# %% [markdown]
# 테스트 데이터에 대한 예측을 수행합니다.

# %%
y_pred_dt = dt_classifier.predict(X_test)

print("Decision Tree 예측 결과 (일부):", y_pred_dt[:10])
print(f"테스트 세트의 실제 사기 거래 수: {sum(y_test)}")
print(f"모델이 예측한 사기 거래 수: {sum(y_pred_dt)}")
print(f"정확도: {accuracy_score(y_test, y_pred_dt):.4f}")

# %% [markdown]
# ### 2.4 Random Forest

# %% [markdown]
# Random Forest 분류기를 생성하고 학습시킵니다. 여러 의사결정 트리를 앙상블하여 더 강건한 모델을 만듭니다.

# %%
rf_classifier = RandomForestClassifier(
    n_estimators=100, 
    max_depth=5, 
    random_state=0, 
)
rf_classifier.fit(X_train, y_train)

# %% [markdown]
# 테스트 데이터에 대한 예측을 수행합니다.

# %%
y_pred_rf = rf_classifier.predict(X_test)

print("Random Forest 예측 결과 (일부):", y_pred_rf[:10])
print(f"테스트 세트의 실제 사기 거래 수: {sum(y_test)}")
print(f"모델이 예측한 사기 거래 수: {sum(y_pred_rf)}")
print(f"정확도: {accuracy_score(y_test, y_pred_rf):.4f}")

# %% [markdown]
# ## 3. 모델 성능 비교

# %% [markdown]
# 각 모델의 성능 지표를 계산하여 비교합니다. 사기 탐지에서는 Recall(재현율)이 특히 중요합니다.

# %%
models = {
    'Logistic Regression': y_pred_lr,
    'KNN': y_pred_knn,
    'Decision Tree': y_pred_dt,
    'Random Forest': y_pred_rf
}

results = []

for model_name, y_pred in models.items():
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('F1 Score', ascending=False)
print("\n=== 모델 성능 비교 ===\n")
print(results_df.to_string(index=False))

# %% [markdown]
# 성능 지표를 시각화하여 비교합니다.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    results_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
    ax.set_title(f'{metric} 비교', fontsize=12, pad=10)
    ax.set_ylabel(metric, fontsize=10)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. 혼동 행렬 비교

# %% [markdown]
# 각 모델의 혼동 행렬을 시각화하여 비교합니다. 사기 탐지에서는 False Negative(사기를 놓치는 경우)를 최소화하는 것이 중요합니다.

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

model_predictions = [
    ('Logistic Regression', y_pred_lr),
    ('KNN', y_pred_knn),
    ('Decision Tree', y_pred_dt),
    ('Random Forest', y_pred_rf)
]

for idx, (model_name, y_pred) in enumerate(model_predictions):
    ax = axes[idx // 2, idx % 2]
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=['Fraud (1)', 'Normal (0)'],
                yticklabels=['Fraud (1)', 'Normal (0)'],
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=12, pad=10)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. 모델별 상세 평가

# %% [markdown]
# 각 모델의 혼동 행렬을 상세히 분석합니다.

# %%
for model_name, y_pred in model_predictions:
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    print(f"\n=== {model_name} ===")
    print(f"Confusion Matrix:\n{cm}")
    print(f"True Positive (TP): {cm[0,0]} - 사기를 사기로 정확히 예측")
    print(f"False Positive (FP): {cm[0,1]} - 정상을 사기로 잘못 예측")
    print(f"False Negative (FN): {cm[1,0]} - 사기를 정상으로 잘못 예측 (위험!)")
    print(f"True Negative (TN): {cm[1,1]} - 정상을 정상으로 정확히 예측")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")

# %% [markdown]
# ## 6. 결론 및 모델 선택

# %% [markdown]
# 각 모델의 특징과 사기 탐지에 대한 적합성을 분석합니다.

# %%
print("\n=== 모델별 특징 및 사기 탐지 적합성 ===\n")

print("1. Logistic Regression:")
print("   - 장점: 빠른 학습 속도, 해석 용이, 선형 관계 학습에 적합")
print("   - 단점: 복잡한 비선형 패턴 학습 어려움")
print("   - 사기 탐지: 기본적인 선형 패턴 탐지에 적합\n")

print("2. KNN:")
print("   - 장점: 간단하고 직관적, 지역 패턴 학습에 강함")
print("   - 단점: 계산 비용이 높고, 고차원 데이터에 취약")
print("   - 사기 탐지: 유사한 거래 패턴 기반 탐지에 적합\n")

print("3. Decision Tree:")
print("   - 장점: 해석 용이, 비선형 관계 학습 가능, 스케일링 불필요")
print("   - 단점: 과적합 위험, 불안정함")
print("   - 사기 탐지: 규칙 기반 탐지에 적합하나 과적합 주의\n")

print("4. Random Forest:")
print("   - 장점: 강건함, 과적합 방지, 비선형 패턴 학습 우수")
print("   - 단점: 해석 어려움, 학습 시간이 상대적으로 김")
print("   - 사기 탐지: 복잡한 패턴 탐지에 가장 적합 (권장)\n")

print("=== 최종 권장 모델 ===")
best_model = results_df.iloc[0]['Model']
print(f"F1 Score 기준 최고 성능 모델: {best_model}")
print(f"사기 탐지에서는 False Negative를 최소화하는 것이 중요하므로,")
print(f"Recall이 높은 모델을 선택하는 것도 고려해야 합니다.")

# %%
