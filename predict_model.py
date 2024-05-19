import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 모델 로드
model = joblib.load('trained_model.pkl')

# 테스트 데이터 로드
X_test = pd.read_parquet('path/to/test_data.parquet.gzip')

# 데이터 전처리
imputer = SimpleImputer(strategy='mean')
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_test_scaled = scaler.transform(X_test_imputed)

# 예측 수행
y_pred_test = model.predict(X_test_scaled)

# 결과 저장
submission = pd.DataFrame({
    'subject_id': X_test['subject_id'],
    'date': X_test['date'],
    'Q1': y_pred_test[:, 0],
    'Q2': y_pred_test[:, 1],
    'Q3': y_pred_test[:, 2],
    'S1': y_pred_test[:, 3],
    'S2': y_pred_test[:, 4],
    'S3': y_pred_test[:, 5],
    'S4': y_pred_test[:, 6],
})
submission.to_csv('predictions.csv', index=False)