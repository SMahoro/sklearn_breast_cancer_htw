import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # algo
from sklearn.metrics import accuracy_score

# 1. Daten laden
breast_cancer = load_breast_cancer()
X = breast_cancer.data        
# Eingabedaten (30 Messwerte)
y = breast_cancer.target      
# Zielwerte (0=malignant, 1=benign)

# 2. Trainings- und Testdaten erzeugen
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,       
    random_state=42,
    stratify=y           
)

# 3. Modell trainieren -> SVM
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)

# 4. Genauigkeit + Vorhersage ausgeben
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Modell-Accuracy (binär): {acc:.3f}")


# 5. Modell + Feature-Namen speichern
joblib.dump(model, "svm_model.pkl")
joblib.dump(breast_cancer.feature_names, "feature_names.pkl")
