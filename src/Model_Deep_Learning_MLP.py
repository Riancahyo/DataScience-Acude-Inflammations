import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd

input_dim = X_train_scaled.shape[1]

# Bangun arsitektur MLP
model_dl = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model_dl.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Training model (> = 10 epoch)
start_time = time.time()

history = model_dl.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=30,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

training_time = time.time() - start_time
print(f"\nTraining time: {training_time:.2f} seconds")

# Evaluasi pada test set
test_loss, test_accuracy = model_dl.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print(f"\nTest Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Prediksi & evaluasi detail
y_pred_prob = model_dl.predict(X_test_scaled)
y_pred_dl = (y_pred_prob >= 0.5).astype(int)

y_test_array = np.array(y_test)

# Gabungkan hasil untuk ditampilkan dalam DataFrame
results = {
    'True Label (y_test)': y_test_array,
    'Predicted Probability': y_pred_prob.flatten(),
    'Final Prediction (y_pred_dl)': y_pred_dl.flatten()
}

df_results = pd.DataFrame(results)
df_results.index.name = 'Test Sample Index'

print("Test Set Predictions (MLP):")
print(df_results)

print("\nClassification Report (MLP):")
print(classification_report(y_test, y_pred_dl))

print("\nConfusion Matrix (MLP):")
cm_dl = confusion_matrix(y_test, y_pred_dl)
disp_dl = ConfusionMatrixDisplay(confusion_matrix=cm_dl, display_labels=[0, 1])
disp_dl.plot()
plt.title('Confusion Matrix - MLP')
plt.show()

# Plot Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Model Summary
model_dl.summary()

# Simpan model ke file .h5
model_dl.save('deep_learning_model.h5')
print("Model Deep Learning (MLP) berhasil disimpan sebagai 'deep_learning_model.h5'.")

