import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

csv_path = Path(__file__).resolve().parents[1] / "notebooks" / "data" / "processed" / "scaled_data.csv"

df = pd.read_csv(csv_path)

x = df.drop(["Will_Buy_EV"],axis=1)
y = df["Will_Buy_EV"]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

lr_model = LogisticRegression()
knn = KNeighborsClassifier(n_neighbors=5)
dt_model = DecisionTreeClassifier(random_state=42)
nb_model = GaussianNB()
svc_model = SVC()

models = [lr_model,knn,dt_model,nb_model,svc_model]

for model in models :
    model.fit(x_train,y_train)

print("Model scores :")
for model in models :
    y_prd  = model.predict(x_test)
    print(f"{model} : ",accuracy_score(y_test,y_prd))
    
    
# the best model comess out ot be the svc_model with the 87.6% accurracy
"""
Model scores :
LogisticRegression() :  0.873
KNeighborsClassifier() :  0.849
DecisionTreeClassifier(random_state=42) :  0.817
GaussianNB() :  0.632
SVC() :  0.876
"""

    

    