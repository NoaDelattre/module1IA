import numpy as np

# --- Le neurone : une fonction qui décide 0 ou 1 ---
def perceptron(entrees, poids, biais):
    z = np.dot(entrees, poids) + biais      # étape 1 : somme pondérée
    if z >= 0:                              # étape 2 : fonction seuil
        return 1                            # étape 3 : décision "activé"
    else:
        return 0                            #           décision "éteint"

# --- On teste notre neurone ---
poids = np.array([0.5, 0.5])    # deux entrées, poids égaux 
biais = -0.2

print(perceptron(np.array([1, 1]), poids, biais))   # 1+1 pondéré = 1.0 - 0.2 = 0.8 >= 0 -> 1
print(perceptron(np.array([0, 0]), poids, biais))   # 0 - 0.2 = -0.2 < 0 -> 0
print(perceptron(np.array([0, 1]), poids, biais))   # 1.0 - 0.2 = 0.8 >= 0 -> 1
print(perceptron(np.array([1, 0]), poids, biais))   # 1.0 - 0.2 = 0.8 >= 0 -> 1