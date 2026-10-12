from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier

data = load_breast_cancer()
X = data.data           # 569 patientes x 30 mesures
y = data.target         # 0 = maligne, 1 = bénigne 

reseau = MLPClassifier(hidden_layer_sizes=(15,), max_iter=2000)
reseau.fit(X, y)

print("Précision :", reseau.score(X, y))

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# MATRICE DE CONFUSION
predictions = reseau.predict(X)
mat = confusion_matrix(y, predictions)

ConfusionMatrixDisplay(mat, display_labels=data.target_names).plot(cmap="Greens")
plt.title("Matrice de confusion - Data")
plt.show()