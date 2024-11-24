from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. 데이터 로드 및 분할
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# 2. 데이터 전처리 (표준화)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. 하이퍼파라미터 튜닝 및 모델 학습
param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 출력
print("최적의 n_neighbors:", grid_search.best_params_['n_neighbors'])
print("교차 검증 평균 정확도:", grid_search.best_score_)

# 4. 테스트 데이터로 최종 평가
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("테스트 데이터 정확도:", accuracy)
#하이퍼파라미터 튜닝을 통해 최적의 k 즉 n_neighbors 값을 결정