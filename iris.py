import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

# 1. Charger les données, garder 2 classes et 2 caractéristiques 
iris = load_iris()
X = iris.data[:100, 2:4]        # 100 premières fleurs, colonnes pétale (longueur, largeur)
y = iris.target[:100]           # étiquettes : 0 = setosa, 1 = versicolor

# 2. Créer et entraîner le perceptron
modele = Perceptron()
modele.fit(X, y)                # le neurone ajuste tout seul ses poids et son biais

# 3. Afficher les fleurs colorées par espèce
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k")
plt.xlabel("Longueur du pétale (cm)")
plt.ylabel("Longueur du pétale (cm)")
plt.title("Perceptron : la frontière qui sépare 2 espèces de fleurs")

# 4. Tracer la frontière de décision (la droite trouvée par le neurone)
w = modele.coef_[0]          # les poids appris
b = modele.intercept_[0]    # le biais appris
x_vals = np.array([X[:, 0].min(), X[:, 0].max()])
y_vals = -(w[0] * x_vals + b / w[1])
plt.plot(x_vals, y_vals, "k--", linewidth=2)

plt.show()