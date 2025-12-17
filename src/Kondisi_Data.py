#@title Cek missing values
import pandas as pd

print("\nCek missing values:")
print(pd.concat([features, target], axis=1).isnull().sum())

#@title Cek duplicate data
print("\nCek duplicate data:")
print(pd.concat([features, target], axis=1).duplicated().sum())

#@title Cek Outliers pada kolom numerik
numerical_columns = ['temperature']

outliers_found = False
outlier_features = []

for col in numerical_columns:
    Q1 = features[col].quantile(0.25)
    Q3 = features[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = features[(features[col] < lower_bound) | (features[col] > upper_bound)]

    if not outliers.empty:
        outliers_found = True
        outlier_features.append(col)

print("Outliers: ", end="")
if outliers_found:
    print(f"Ada, pada fitur: {', '.join(outlier_features)}")
else:
    print("Tidak ada")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.boxplot(y=features['temperature'])
plt.title('Box Plot of Temperature')
plt.ylabel('Temperature (°C)')
plt.show()

#@title Cek Imbalanced Data
imbalanced_status = []
ratio_details = []

col = target.name
value_counts = target.value_counts()
if len(value_counts) > 1:
    majority_class = value_counts.max()
    minority_class = value_counts.min()
    ratio = round(majority_class / minority_class, 1) if minority_class > 0 else float('inf')
    ratio_details.append(f"{col} ({ratio}:1)")

    if ratio >= 1.5:
        imbalanced_status.append("Ada")
    else:
        imbalanced_status.append("Tidak ada")
else:
    ratio_details.append(f"{col} (Hanya satu kelas)")
    imbalanced_status.append("Tidak ada")

final_imbalance_status = "Ada" if "Ada" in imbalanced_status else "Tidak ada"
final_ratio_detail = ', '.join(ratio_details)

print(f"Imbalanced Data: {final_imbalance_status}, rasio kelas: {final_ratio_detail}")

#@title Cek Noise
print("\n--- Cek Noise pada Kolom Fitur Kategorikal ---")
for col in features.select_dtypes(include='object').columns:
    print(f"\nKolom '{col}':")
    display(features[col].value_counts())

print("\n--- Cek Noise pada Kolom Target Kategorikal ---")
print(f"\nKolom '{target.name}':")
display(target.value_counts())

