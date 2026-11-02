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