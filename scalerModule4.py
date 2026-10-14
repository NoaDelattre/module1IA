from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)     # calcule l'échelle SUR le train et l'applique
X_test = scaler.transform(X_test)           # applique la MÊME échelle au test
