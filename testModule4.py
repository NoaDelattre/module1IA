from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 

train = train_test_split()
X = train.data
y = train.target
reseau = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500)
reseau.fit(X, y)

# X = données, y = étiquettes
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20 % réservés au test
    random_state=42     # pour obtenir le même découpage à chaque exécution
)

reseau.fit(X_train, y_train)            # on entraîne SEULEMENT sur le train
print("Score test :", reseau.score(X_test, y_test))     # on évalue sur le test