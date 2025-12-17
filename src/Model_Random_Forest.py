from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
import joblib

# Inisialisasi model Random Forest
model_advanced = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Training model dan hitung waktu training
start_time_rf = time.time()
model_advanced.fit(X_train_scaled, y_train)
end_time_rf = time.time()
training_time_rf = end_time_rf - start_time_rf
print(f"Training time Random Forest: {training_time_rf:.4f} seconds")

# Prediksi pada data testing
y_pred_advanced = model_advanced.predict(X_test_scaled)
y_pred_proba_advanced = model_advanced.predict_proba(X_test_scaled)[:, 1]

# Evaluasi model
accuracy_advanced = accuracy_score(y_test, y_pred_advanced)
roc_auc_advanced = roc_auc_score(y_test, y_pred_proba_advanced)
print(f"Accuracy Random Forest: {accuracy_advanced:.4f}")
print(f"ROC-AUC Random Forest: {roc_auc_advanced:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_advanced))

print("\nConfusion Matrix:")
cm_advanced = confusion_matrix(y_test, y_pred_advanced)
disp_advanced = ConfusionMatrixDisplay(confusion_matrix=cm_advanced, display_labels=model_advanced.classes_)
disp_advanced.plot()
plt.title('Confusion Matrix - Random Forest')
plt.show()

# Simpan model ke file .pkl
joblib.dump(model_advanced, 'random_forest_model.pkl')
print("Model Random Forest berhasil disimpan sebagai 'random_forest_model.pkl'.")

