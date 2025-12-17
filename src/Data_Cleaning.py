#@title Remove duplicate data
print(f"Jumlah duplikat sebelum dihapus: {pd.concat([features, target], axis=1).duplicated().sum()}")

data_combined = pd.concat([features, target], axis=1)
data_combined_cleaned = data_combined.drop_duplicates()

features_cleaned = data_combined_cleaned[features.columns]
targets_cleaned = data_combined_cleaned[[target.name]]

print("Duplikat telah dihapus.")
print(f"Bentuk fitur setelah dihapus duplikat: {features_cleaned.shape}")
print(f"Bentuk target setelah dihapus duplikat: {targets_cleaned.shape}")

#@title Data Type Conversion

# Daftar fitur biner
binary_feature_columns = [
    'nausea',
    'lumbar-pain',
    'urine-pushing',
    'micturition-pains',
    'burning-urethra'
]

for col in binary_feature_columns:
    features_cleaned[col] = features_cleaned[col].replace({'yes': 1, 'no': 0})
# Pastikan fitur biner bertipe integer
features_cleaned[binary_feature_columns] = features_cleaned[binary_feature_columns].astype(int)

target_data = targets_cleaned.replace({'yes': 1, 'no': 0}).astype(int)

# mengelompokkan kembali binary fitur column dengan temperature
features_data = binary_feature_columns + ['temperature']

