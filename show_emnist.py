import matplotlib.pyplot as plt
from emnist import extract_training_samples
import numpy as np

DATASET = 'balanced'
X, y = extract_training_samples(DATASET)

# Mapa de clases (igual que tú usas)
import string
digits = {i: str(i) for i in range(10)}
letters = {i+10: c for i, c in enumerate(string.ascii_uppercase)}
class_map = {**digits, **letters}

# Muestra ejemplos de una clase concreta
def show_class(label, n=16):
    idx = np.where(y == label)[0][:n]
    fig, axes = plt.subplots(4, 4, figsize=(4,4))
    for ax, i in zip(axes.flat, idx):
        ax.imshow(X[i], cmap='gray')
        ax.axis('off')
    plt.suptitle(f"Class {label} -> {class_map.get(label)}")
    plt.show()

# Ejemplos problemáticos
show_class(1)   # dígito 1
show_class(18)  # I
show_class(19)  # J
show_class(5)   # dígito 5
show_class(9)   # dígito 9
