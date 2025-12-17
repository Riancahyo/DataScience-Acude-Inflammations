#@title Scaling
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Inisialisasi StandardScaler
scaler = StandardScaler()

# Skala fitur pada data training
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)

# Skala fitur pada data validasi dan testing menggunakan scaler yang sudah di-fit pada data training
X_val_scaled = scaler.transform(X_val)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)

X_test_scaled = scaler.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

print("Data fitur setelah scaling:")
display(X_train_scaled.head())

