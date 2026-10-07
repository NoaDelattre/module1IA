import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

predictions = reseau.predict(X)
mat = confusion_matrix(y, predictions)

ConfusionMatrixDisplay(mat, display_labels=iris.target_names).plot(cmap="Greens")
plt.title("Matrice de confusion - Iris")
plt.show()