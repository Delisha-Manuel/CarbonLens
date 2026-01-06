import pandas as pd
import ast
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("data/Carbon Emission.csv")

print(df.head())
print(df.info())
print(df.columns.tolist())

features = df.drop(columns=['CarbonEmission', 'Body Type', 'Sex'])
target = df['CarbonEmission']

print(features.head())
print(target.head())

features['Vehicle Type'] = features['Vehicle Type'].fillna("Unknown")

features['Recycling_Count'] = features['Recycling'].apply(lambda x: len(ast.literal_eval(x)))
features['Cooking_Count'] = features['Cooking_With'].apply(lambda x: len(ast.literal_eval(x)))
features = features.drop(columns=['Recycling', 'Cooking_With'])

categorical_cols = features.select_dtypes(include='object').columns
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    le.fit(list(features[col]) + ["Unknown"])
    features[col] = le.transform(features[col])
    le_dict[col] = le

features_train, features_test, target_train, target_test = train_test_split(features, target, random_state = 17, test_size = 0.2)

rf = RandomForestRegressor(n_estimators = 200, random_state = 17, n_jobs = -1)

rf.fit(features_train, target_train)

target_pred = rf.predict(features_test)
rmse = ((target_test - target_pred) ** 2).mean() ** 0.5
r2 = rf.score(features_test, target_test)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")

importances = rf.feature_importances_
feature_names = features_train.columns

feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
feat_imp = feat_imp.sort_values(by = 'importance', ascending = False)

print("\nTop 10 most important features:")
print(feat_imp.head(10))

top_10 = feat_imp.head(10)

plt.figure(figsize = (8,6))
plt.barh(top_10['feature'][::-1], top_10['importance'][::-1], color='teal')
plt.xlabel("Feature Importance")
plt.title("Top 10 Features Affecting Carbon Emission")
plt.tight_layout()
plt.show()

with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("le_dict.pkl", "wb") as f:
    pickle.dump(le_dict, f)

print("DEBUG: Model and encoders saved!")
