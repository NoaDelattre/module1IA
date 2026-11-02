import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

#1. CHARGER
df = pd.read_csv("./titanic/titanic.csv")
print(df.head())

# 2. PRÉTRAITER (les données réelles sont sales !)
df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]       # colonnes utiles
df["Age"] = df["Age"].fillna(df["Age"].median())            # remplir les âges manquants
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})         # texte -> nombre
df = df.dropna()                                            # retirer le reste des trous

# 3. SÉPARER entrées (X) et cible (y)
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. NORMALISER
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. ENTRAÎNER puis ÉVALUER sur des passagers jamais vus
reseau = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000)
reseau.fit(X_train, y_train)
print("Score test :", reseau.score(X_test, y_test))     # typiquement ~0.78 à 0.82



from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

predictions = reseau.predict(X_test)

# Rapport texte : precision, recall, f1 pour chaque classe
print(classification_report(y_test, predictions))

# Matrice de confusion visuelle (heatmap seaborn)
mat = confusion_matrix(y_test, predictions)
sns.heatmap(mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Décédé", "Survivant"],
            yticklabels=["Décédé", "Survivant"])
plt.xlabel("Prédiction du modèle")
plt.ylabel("Réalité")
plt.title("Fiche de recette - matrice de confusion Titanic")
plt.show()