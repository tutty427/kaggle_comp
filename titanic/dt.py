import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn import preprocessing
from scipy.stats import mode
import warnings

from sklearn.feature_selection import RFECV
# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

warnings.filterwarnings('ignore')

url = "D:/workspace_py/pyproject/titanic/train.csv"
defaultTrainData = pd.read_csv(url)


testUrl = "D:/workspace_py/pyproject/titanic/test.csv"
defaultTestData = pd.read_csv(testUrl)



embarkedLe = preprocessing.LabelEncoder()
embarkedLe.fit(defaultTrainData['Embarked'])


sexLe = preprocessing.LabelEncoder()
sexLe.fit(defaultTrainData['Sex'])


def num_missing(x):
    return sum(x.isnull())

def clearData(x,isTest):
    if isTest:
        cleanTrainData = x.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
    else:
        cleanTrainData = x.drop(['PassengerId','Name','Ticket','Cabin','Survived'],axis=1)

    naV = cleanTrainData.apply(num_missing,axis=0);
    index = 0
    for val in naV:
        if val > 0:
            colName = cleanTrainData.columns.values[index]
            cleanTrainData[colName].fillna(mode(cleanTrainData[colName]).mode[0],inplace=True)
        index+=1

    cleanTrainData.loc[:,'Embarked'] = embarkedLe.transform(cleanTrainData['Embarked'])
    cleanTrainData.loc[:,'Sex'] = sexLe.transform(cleanTrainData['Sex'])

    return cleanTrainData


def decisionTreeModel(X,Y):
    num_trees = 100
    max_features = 7
    kfold = KFold(n_splits=10, random_state=7)
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    # results = cross_val_score(model, X, Y, cv=kfold)
    # return results
    model.fit(X,Y)
    return model


def logisticRegression(X,Y):
    model = LogisticRegression()
    model.fit(X,Y)
    return model


def SVCM(X,Y):
    model = SVC()
    model.fit(X,Y)
    return model

def train():

    X = clearData(defaultTrainData,False).values;
    Y = defaultTrainData['Survived'].values;
    train_X , valid_X , train_y , valid_y = train_test_split( X , Y , train_size = .7 )

    modelA =  decisionTreeModel(train_X,train_y)
    print modelA.score( train_X,train_y)
    print modelA.score( valid_X,valid_y)


    # modelB = logisticRegression(train_X,train_y)
    # print modelB.score( train_X,train_y)
    # print modelB.score( valid_X,valid_y)
    #
    #
    # modelC = SVCM(train_X,train_y)
    # print modelC.score( train_X,train_y)
    # print modelC.score( valid_X,valid_y)

    test_X =clearData(defaultTestData,True).values
    test_Y = modelA.predict(test_X)

    pId = defaultTestData['PassengerId']

    result = pd.DataFrame({'PassengerId':pId,'Survived':test_Y})
    result.to_csv("D:/workspace_py/pyproject/titanic/result.csv",index=False)


print train()