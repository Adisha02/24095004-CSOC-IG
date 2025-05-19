#i have taken the same script for all but for part 1 i have to clean some part of data as it contains string there , which i did with the help of chatgpt

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

df=pd.read_csv("housing.csv")
X=df.drop(df.columns[[-2,-1]],axis=1)
y=df["median_house_value"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)

# cleaning the data

print("NaN in X_train:", np.isnan(X_train).any())
print("Inf in X_train:", np.isinf(X_train).any())
print("NaN in y_train:", np.isnan(y_train).any())
print("Inf in y_train:", np.isinf(y_train).any())

# Remove rows with NaNs or Infs
mask = ~np.isnan(X_train).any(axis=1)  & ~np.isnan(y_train)
X_train = X_train[mask]
y_train = y_train[mask]

gdr = Gdregressor(epochs=50, learning_rate=0.2)


#for part 1
def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def clean_data(X):
    cleaned = []
    for row in X:
        cleaned_row = [float(x) for x in row if is_number(x)]
        cleaned.append(cleaned_row)
    return cleaned

X_train_cleaned = clean_data(X_train)
X_test_cleaned = clean_data(X_test)

gdr.fit(X_train_cleaned, y_train)
y_pred = gdr.predict(X_test_cleaned)
print("Intercept:", gdr.intercept_)
print("Coefficients:", gdr.coef_)

#for part 2 and 3
gdr = Gdregressor(epochs=50, learning_rate=0.2)
gdr.fit(X_train,y_train)
print("Intercept:", gdr.intercept_)
print("Coefficients:", gdr.coef_)

import numpy as np
from sklearn.metrics import r2_score

# Convert to NumPy arrays if not already
y_test = np.array(y_test)
y_pred = np.array(y_pred)

# Check for NaNs
mask = ~np.isnan(y_test) & ~np.isnan(y_pred)

# Use only valid values
y_test_clean = y_test[mask]
y_pred_clean = y_pred[mask]

# Now calculate R²
r2 = r2_score(y_test_clean, y_pred_clean)
print("R² Score:", r2)
