import joblib
import numpy as np

#model laden
model=joblib.load("model.pkl")

#beispiel-features
features=[5.1,3.5,1.4,0.2]

#in richitge form brungen
X=np.array(features).reshape(1,-1)

#vorhersage machen
prediction =model.predict(X)[0]
probabiliies=model.predict_proba(X)[0]

#Ausgabe
print("Prediction:", prediction)
print("Propabilities:". probabilites)