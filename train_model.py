import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# 데이터 로드
gps = pd.read_parquet('path/to/gps.parquet.gzip')
ambience = pd.read_parquet('path/to/ambience.parquet.gzip')
activity = pd.read_parquet('path/to/activity.parquet.gzip')
m_light = pd.read_parquet('path/to/m_light.parquet.gzip')
usage_stats = pd.read_parquet('path/to/usage_stats.parquet.gzip')
heart_rate = pd.read_parquet('path/to/heart_rate.parquet.gzip')
w_light = pd.read_parquet('path/to/w_light.parquet.gzip')
pedo = pd.read_parquet('path/to/pedo.parquet.gzip')
train_label = pd.read_csv('path/to/train_label.csv')

# 데이터 병합
dataframes = [gps, ambience, activity, m_light, usage_stats, heart_rate, w_light, pedo]
merged_data = pd.concat(dataframes, axis=1)
merged_data = pd.merge(merged_data, train_label, on=['subject_id', 'date'], how='left')

# 데이터 전처리
imputer = SimpleImputer(strategy='mean')
merged_data_imputed = imputer.fit_transform(merged_data)

scaler = StandardScaler()
merged_data_scaled = scaler.fit_transform(merged_data_imputed)

# 특성과 타겟 분리
X = merged_data_scaled[:, :-7]
y = merged_data_scaled[:, -7:]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=2, random_state=42)
model.fit(X_train, y_train)

# 모델 저장
joblib.dump(model, 'trained_model.pkl')