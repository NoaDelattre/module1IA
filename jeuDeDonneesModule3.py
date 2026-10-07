from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier

data = load_breast_cancer()
X = data.data           # 569 patientes x 30 mesures
y = data.target         # 0 = maligne, 1 = bénigne 

reseau = MLPClassifier(hidden_layer_sizes=(15,), max_iter=2000)
reseau.fit(X, y)

print("Précision :", reseau.score(X, y))