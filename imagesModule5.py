import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Charger MNIST (téléchargé automatiquement la 1re fois)
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("Images d'entraînement :", X_train.shape)   # (60000, 28, 28)

# Afficher la 1re image et son étiquette
plt.imshow(X_train[0], cmap="gray")
plt.title("Étiquette réelle : " + str(y_train[0]))
plt.show()

# 1. Normaliser : ramener les pixels de 0-255 vers 0-1 (Le réseau préfère)
X_train = X_train / 255.0
X_test = X_test / 255.0 

# Construire le réseau : on empile les couches
modele = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),   # aplatit la grille 28x28 en 784 entrées
    layers.Dense(128, activation="relu"),   # couche cachée de 128 neurones
    layers.Dense(64, activation="relu"),    # couche cachée de 64 neurones
    layers.Dense(10, activation="softmax")  # 10 neurones de sortie (un par chiffre 0-9)
])

# Configurer l'entraînement
modele.compile(optimizer="adam",
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])

# Afficher l'architecture
modele.summary()

# Entraîner : 5 passages (epochs) sur les données
historique = modele.fit(X_train, y_train,
                        epochs=5,
                        validation_split=0.1)   # garde 10% pour surveiler l'overfitting

# Évaluer sur les données de test (jamais vues)
perte, precision = modele.evaluate(X_test, y_test)
print("Précision sur le test :", precision)     # typiquement ~0.97 -> 97 % !

# Prédictions sur le jeu de test
predictions = modele.predict(X_test)
chiffres_predits = np.argmax(predictions, axis=1)   # le chiffre le plus probable

# Trouver les indices où le réseau s'est trompé
erreurs = np.where(chiffres_predits != y_test)[0]
print("Nombre d'erreurs :", len(erreurs))

# Afficher la 1re erreur
i = erreurs[0]
plt.imshow(X_test[i], cmap="gray")
plt.title("Réel : " + str(y_test[i]) + " | Prédit : " + str(chiffres_predits[i]))
plt.show()