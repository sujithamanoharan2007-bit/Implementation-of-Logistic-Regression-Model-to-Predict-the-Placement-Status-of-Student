 # Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results. 
## Program:
```
DEVELOPED BY:SUJITHA MAHALAKSHMI M
REF NO: 25018945

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Placement_Data.csv")   

print("Dataset Preview:")
print(data.head())

data = data.drop(["sl_no", "salary"], axis=1)

data["status"] = data["status"].map({"Placed": 1, "Not Placed": 0})

X = data.drop("status", axis=1)
y = data["status"]

X = pd.get_dummies(X, drop_first=True)

print("\nAfter Encoding:")
print(X.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Placement Prediction")
plt.show()
```
## Output:
```
Dataset Preview:
   sl_no gender  ssc_p    ssc_b  hsc_p    hsc_b     hsc_s  degree_p  \
0      1      M  67.00   Others  91.00   Others  Commerce     58.00   
1      2      M  79.33  Central  78.33   Others   Science     77.48   
2      3      M  65.00  Central  68.00  Central      Arts     64.00   
3      4      M  56.00  Central  52.00  Central   Science     52.00   
4      5      M  85.80  Central  73.60  Central  Commerce     73.30   

    degree_t workex  etest_p specialisation  mba_p      status    salary  
0   Sci&Tech     No     55.0         Mkt&HR  58.80      Placed  270000.0  
1   Sci&Tech    Yes     86.5        Mkt&Fin  66.28      Placed  200000.0  
2  Comm&Mgmt     No     75.0        Mkt&Fin  57.80      Placed  250000.0  
3   Sci&Tech     No     66.0         Mkt&HR  59.43  Not Placed       NaN  
4  Comm&Mgmt     No     96.8        Mkt&Fin  55.50      Placed  425000.0  

After Encoding:
   ssc_p  hsc_p  degree_p  etest_p  mba_p  gender_M  ssc_b_Others  \
0  67.00  91.00     58.00     55.0  58.80         1             1   
1  79.33  78.33     77.48     86.5  66.28         1             0   
2  65.00  68.00     64.00     75.0  57.80         1             0   
3  56.00  52.00     52.00     66.0  59.43         1             0   
4  85.80  73.60     73.30     96.8  55.50         1             0   

   hsc_b_Others  hsc_s_Commerce  hsc_s_Science  degree_t_Others  \
0             1               1              0                0   
1             1               0              1                0   
2             0               0              0                0   
3             0               0              1                0   
4             0               1              0                0   

   degree_t_Sci&Tech  workex_Yes  specialisation_Mkt&HR  
0                  1           0                      1  
1                  1           1                      0  
2                  0           0                      0  
3                  1           0                      1  
4                  0           0                      0  

Accuracy: 0.8837209302325582

Classification Report:
              precision    recall  f1-score   support

           0       0.82      0.75      0.78        12
           1       0.91      0.94      0.92        31

    accuracy                           0.88        43
   macro avg       0.86      0.84      0.85        43
weighted avg       0.88      0.88      0.88        43
```
<img width="900" height="601" alt="image" src="https://github.com/user-attachments/assets/de137b40-3845-4161-987f-00facc63c6a9" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
