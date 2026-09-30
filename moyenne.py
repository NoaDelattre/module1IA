notes = [16, 18, 17]
somme = 0

for n in range(0,len(notes)):
    somme = somme + notes[n]
    n = n + 1
moyenne = somme/len(notes)
if moyenne >= 10:
    print("Admis")
else:
    print("Rattrapage nécessaire")
print("Moyenne : ", moyenne) 