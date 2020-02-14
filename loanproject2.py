import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create dataframe, drop null values
df = pd.read_csv("/Users/steven/datasets/trainloans.csv")
df.dropna(how = 'any', inplace = True)

# Label encode target variable
le = LabelEncoder()
df["Loan_Status"] = le.fit_transform(df["Loan_Status"])

# Split arrays or matrices into random train and test subsets
train, test = train_test_split(df)

# Identify target variable (status) and independent variable (ID)
X_train = train.drop(columns = ["Loan_ID","Loan_Status"], axis = 1)
y_train = train["Loan_Status"]

X_test = test.drop(columns = ["Loan_ID","Loan_Status"], axis = 1)
y_test = test["Loan_Status"]

# Encode data
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

#Create model, print result
def Regression():
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
   
    print("Prediction: ", predict)
    print("Actual: ", y_test)
    print("Accuracy Score: ", accuracy_score(y_test, predict))


Regression()
