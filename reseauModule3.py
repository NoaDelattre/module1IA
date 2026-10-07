from sklearn.datasets import load_iris 
from sklearn.neural_network import MLPClassifier 

# 1. DONNEES : X = les mesures, y = l'espece a deviner
iris = load_iris()
X = iris.data               # 150 fleurs x 4 mesures
y = iris.target             # 150 étiquettes (0, 1 ou 2)

# 2. MODELE : un reseau avec UNE couche cachee de 10 neurones
reseau = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000)

# 3. ENTRAINEMENT : le reseau ajuste tous ses poids automatiquement
reseau.fit(X, y)

# 4. EVALUATION : quel pourcentage de fleurs bien classees ?
score = reseau.score(X, y)
print("Précision :", score)         # ex. : 0.98 -> 98% de bonnes réponses !

# 5. PREDICTION sur une nouvelle fleur jamais vue
nouvelle_fleur = [[5.1, 3.5, 1.4, 0.2]]
print("Espèce prédite :", reseau.predict(nouvelle_fleur))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# MATRICE DE CONFUSION
predictions = reseau.predict(X)
mat = confusion_matrix(y, predictions)

ConfusionMatrixDisplay(mat, display_labels=iris.target_names).plot(cmap="Greens")
plt.title("Matrice de confusion - Iris")
plt.show()