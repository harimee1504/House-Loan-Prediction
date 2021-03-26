from flask import Flask, render_template, redirect, request, session, url_for
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from random import randint
from csv import writer
import seaborn as sns
import pandas as pd
import numpy as np
import random

app = Flask(__name__)

app.secret_key = "Your Password Here"

app.config["DEBUG"] = True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/details')
def details():
    return render_template('details.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect('/')
    else:
        name = request.form["name"]
        loanid = 'LP'+str(random.randint(100000, 999999))
        gender = request.form["gender"]
        married = request.form["married"]
        dependents = request.form["dependents"]
        education = request.form["education"]
        selfemployed = request.form["selfemployed"]
        applicantincome = request.form["applicantincome"]
        coapplicantincome = request.form["coapplicantincome"]
        loanamount = request.form["loanamount"]
        loanamountterm = request.form["loanamountterm"]
        credit = request.form["credit"]
        area = request.form["area"]
        lst = [loanid, gender, married, dependents, education, selfemployed, applicantincome,
               coapplicantincome, loanamount, loanamountterm, credit, area]

        with open('./static/Dataset/test.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(lst)
            f_object.close()
        session['user_name'] = name
        return redirect(url_for('test'))


@app.route('/test')
def test():
    train = pd.read_csv('./static/Dataset/train.csv')
    train.Loan_Status = train.Loan_Status.map({'Y': 1, 'N': 0})

    Loan_status = train.Loan_Status
    train.drop('Loan_Status', axis=1, inplace=True)
    test = pd.read_csv('./static/Dataset/test.csv')
    Loan_ID = test.Loan_ID
    data = train.append(test)

    data.Gender = data.Gender.map({'Male': 1, 'Female': 0})

    data.Married = data.Married.map({'Yes': 1, 'No': 0})

    data.Dependents = data.Dependents.map({'0': 0, '1': 1, '2': 2, '3+': 3})

    data.Education = data.Education.map({'Graduate': 1, 'Not Graduate': 0})

    data.Self_Employed = data.Self_Employed.map({'Yes': 1, 'No': 0})

    data.Property_Area = data.Property_Area.map(
        {'Urban': 2, 'Rural': 0, 'Semiurban': 1})

    data.Credit_History.fillna(np.random.randint(0, 2), inplace=True)

    data.Married.fillna(np.random.randint(0, 2), inplace=True)

    data.LoanAmount.fillna(data.LoanAmount.median(), inplace=True)

    data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(), inplace=True)

    data.Gender.fillna(np.random.randint(0, 2), inplace=True)

    data.Dependents.fillna(data.Dependents.median(), inplace=True)

    data.Self_Employed.fillna(np.random.randint(0, 2), inplace=True)

    data.drop('Loan_ID', inplace=True, axis=1)

    train_X = data.iloc[:614, ]
    train_y = Loan_status
    X_test = data.iloc[614:, ]
    seed = 7

    train_X, test_X, train_y, test_y = train_test_split(
        train_X, train_y, random_state=seed)

    models = []
    models.append(("logreg", LogisticRegression()))
    models.append(("tree", DecisionTreeClassifier()))
    models.append(("lda", LinearDiscriminantAnalysis()))
    models.append(("svc", SVC()))
    models.append(("knn", KNeighborsClassifier()))
    models.append(("nb", GaussianNB()))

    seed = 7
    scoring = 'accuracy'

    result = []
    names = []

    for name, model in models:
        kfold = KFold(n_splits=10, random_state=seed)
        cv_result = cross_val_score(
            model, train_X, train_y, cv=kfold, scoring=scoring)
        result.append(cv_result)
        names.append(name)

    svc = LogisticRegression()
    svc.fit(train_X, train_y)
    pred = svc.predict(test_X)

    df_output = pd.DataFrame()

    outp = svc.predict(X_test).astype(int)

    df_output['Loan_ID'] = Loan_ID
    df_output['Loan_Status'] = outp

    result = df_output['Loan_Status'].tolist()

    res = str(result[-1])
    user_name = session.get('user_name', None)
    if res == '1':
        msg = "Congratulations ! You are Eligible to Get Loan."
    else:
        msg = "Sorry , You are Not Eligible to Get Loan."
    return render_template('output.html', msg=msg, name=user_name)


if __name__ == '__main__':
    app.run(debug=True)
