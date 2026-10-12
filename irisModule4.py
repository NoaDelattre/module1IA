import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris 

iris = load_iris()
reseau = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500)
reseau.fit(iris.data, iris.target)

# La courbe de loss : on doit voir l'erreur DESCENDRE epoch après epoch
plt.plot(reseau.loss_curve_)
plt.xlabel("Epochs (passages sur les données)")
plt.ylabel("Loss (erreur)")
plt.title("Courbe d'apprentissage : le réseau progresse !")
plt.show()
