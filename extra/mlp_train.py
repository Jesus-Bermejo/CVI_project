import numpy as np
from emnist import extract_training_samples, extract_test_samples
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# =========================
# 1. Cargar EMNIST
# =========================
DATASET = 'balanced'

print("Loading EMNIST:", DATASET)
X_train, y_train = extract_training_samples(DATASET)
X_test, y_test   = extract_test_samples(DATASET)

print("Train samples:", X_train.shape)
print("Test samples:", X_test.shape)

# =========================
# 2. Filtrado de clases (opcional)
# =========================
# Filtrar solo A-Z y 0-9 si quieres
allowed_classes = list(range(10)) + list(range(10, 36))  # 0-9 y A-Z según balanced
train_mask = np.isin(y_train, allowed_classes)
test_mask  = np.isin(y_test, allowed_classes)

X_train, y_train = X_train[train_mask], y_train[train_mask]
X_test, y_test   = X_test[test_mask], y_test[test_mask]

# =========================
# 3. Preprocesado
# =========================
X_train = X_train.astype(np.float32) / 255.0
X_test  = X_test.astype(np.float32) / 255.0

X_train = X_train.reshape(len(X_train), -1)
X_test  = X_test.reshape(len(X_test), -1)

# =========================
# 4. Modelo mejorado
# =========================
model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    max_iter=50,
    batch_size=256,
    verbose=True,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# =========================
# 5. Evaluación
# =========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

# =========================
# 6. Guardar modelo
# =========================
joblib.dump(model, "char_mlp.pkl")
print("Model saved as char_mlp.pkl")
