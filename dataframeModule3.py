import pandas as pd
from sklearn.datasets import load_iris

# Charger Iris et le transformer en tableau pandas lisible
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["espece"] = iris.target      # ajoute une colonne avec l'étiquette (0, 1, 2)

print(df.head())        # affiche les 5 premières lignes
print(df.describe())    # statistiques : moyenne, min, max, écart-type...