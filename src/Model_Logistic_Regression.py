from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import joblib

# Inisialisasi model Logistic Regression
model_baseline = LogisticRegression(
    C=1.0,
    solver='lbfgs',
    max_iter=100,
    random_state=42
)

# Training model dan hitung waktu training
start_time_lr = time.time()
model_baseline.fit(X_train_scaled, y_train)
end_time_lr = time.time()
training_time_lr = end_time_lr - start_time_lr
print(f"Training time Logistic Regression: {training_time_lr:.4f} seconds")

# Prediksi pada data testing
y_pred_baseline = model_baseline.predict(X_test_scaled)
y_pred_proba_baseline = model_baseline.predict_proba(X_test_scaled)[:, 1]

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred_baseline)
roc_auc = roc_auc_score(y_test, y_pred_proba_baseline)
print(f"Accuracy Logistic Regression: {accuracy:.4f}")
print(f"ROC-AUC Logistic Regression: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_baseline))

print("\nConfusion Matrix:")
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
disp_baseline = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=model_baseline.classes_)
disp_baseline.plot()
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Simpan model ke file .pkl
joblib.dump(model_baseline, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model Logistic Regression dan scaler berhasil disimpan sebagai 'logistic_regression_model.pkl' dan 'scaler.pkl'.")

