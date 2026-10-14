from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split 
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. CHARGER
df = pd.read_csv("titanic.csv")
print(df.head())

# 2. PRÉTRAITER (les données réelles sont sales !)
df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]   # colonnes utiles
df["Age"] = df["Age"].fillna(df["Age"].median())        # remplir les âges manquants
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})     # texte -> nombre
df = df.dropna()                                        # retirer le reste des trous

# 3. SÉPARER entrées (X) et cible (y)
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]


reseau = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500)

# X = données, y = étiquettes
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20 % réservés au test
    random_state=42     # pour obtenir le même découpage à chaque exécution
)

reseau.fit(X_train, y_train)            # on entraîne SEULEMENT sur le train
print("Score test :", reseau.score(X_test, y_test))     # on évalue sur le test

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)     # calcule l'échelle SUR le train et l'applique
X_test = scaler.transform(X_test)           # applique la MÊME échelle au test