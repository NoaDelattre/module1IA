# Mon tout premier programme Python pour l'IA !
# Le # introduit un commentaire : Python l'ignore complètement.
# Les commentaires servent à expliquer le code - prends l'habitude d'en écrire !

print("Bonjour l'IA !")
print("Tout commence ici.")

# Variables

prenom  = "Alice"           # str : chaîne de caractères (entre guillemets)
age     = 20                # int : nombre entier
note    = 15.5              # float : nombre décimal (avec un point, pas de virgule !)
admis   = True              # bool : vrai (True) ou faux (False)

print(prenom, "a", age, "ans - note :", note, "- admis :", admis)

notes = [14, 18, 12, 16, 9]     # liste de 5 entiers

print(notes)                # affiche [14, 18, 12, 16, 9]
print(notes[0])             # affiche 14    (le 1er élément - indice 0)
print(notes[4])             # affiche 9     (le 5e élément - indice 4)
print(len(notes))           # affiche 5     (nombre d'éléments dans la liste)

for n in notes:
    print("Note :", n)      # <-- ce bloc est décalé de 4 espaces : c'est l'indentation

note1 = 14

if note1 >= 10:
    print("Admis")          # <-- bloc exécuté si la condition est vraie
else:
    print("Ajourné")        # <-- bloc exécuté si la condition est fausse