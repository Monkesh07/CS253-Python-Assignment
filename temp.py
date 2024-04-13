import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

def int_converter(value):
    if not isinstance(value, str):
        value = str(value)
    if isinstance(value, str):
        if 'Crore' in value:
            return float(value.replace(' Crore+', ''))*10000
        elif 'Lac' in value:
            return float(value.replace(' Lac+', ''))*100
        elif 'Thou' in value:
            return float(value.replace(' Thou+', ''))
        elif 'Hund' in value:
            return float(value.replace(' Hund+', ''))/10
        else:
            return float(value) / 1000
    return value/1000

train_df = pd.read_csv('/kaggle/input/who-is-the-real-winner/train.csv')
train_df.info()

data = pd.read_csv('/kaggle/input/who-is-the-real-winner/train.csv')
finalTest = pd.read_csv('/kaggle/input/who-is-the-real-winner/test.csv')

label_encoders = {}
for column in ["Party", "state"]:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])
    finalTest[column] = label_encoders[column].transform(finalTest[column])

label_encoders["Education"] = LabelEncoder()
data["Education"] = label_encoders["Education"].fit_transform(data["Education"])

X = data[["Party", "Criminal Case", "Total Assets", "Liabilities", "state"]]
y = data["Education"]
finalTestX = finalTest[["Party", "Criminal Case", "Total Assets", "Liabilities", "state"]]

for col in ['Total Assets', 'Liabilities']:
    X[col] = X[col].apply(int_converter)
    finalTestX[col] = finalTestX[col].apply(int_converter)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=81)
rf_classifier.fit(X_train, y_train)
predictions = rf_classifier.predict(X_test)
predictions = label_encoders["Education"].inverse_transform(predictions)
accuracy = (predictions == label_encoders["Education"].inverse_transform(y_test)).mean()
print("Accuracy:", accuracy)

f1_score(label_encoders["Education"].inverse_transform(y_test), predictions, average='weighted')

rf_classifier.fit(X, y)
predictions = rf_classifier.predict(X)
predictions = label_encoders["Education"].inverse_transform(predictions)
accuracy = (predictions == label_encoders["Education"].inverse_transform(y)).mean()
print("Accuracy:", accuracy)

predictions = rf_classifier.predict(finalTestX)
predictions = label_encoders["Education"].inverse_transform(predictions)
submission_df = pd.DataFrame({'ID': finalTest['ID'], 'Education': predictions})
submission_df.to_csv('my_submission5.csv', index=False)
