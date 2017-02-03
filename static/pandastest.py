import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
# dates = pd.date_range('20130101', periods=6)
# print(dates)
#
# df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
# print(df)
#
# df2 = pd.DataFrame({'A':1.,'B':pd.Timestamp('20170614'),'C': pd.Series(1,index=list(range(4)),dtype='float32'),'D':np.array([3] * 4,dtype='int32'),'E':pd.Categorical(["test","train","test","train"]),'F':'foo'});
# print(df2)
#
# print(df2.dtypes)
#
# print(df2.head(2))
# arr = np.array([[1,2,3],[4,5,6]])
# rowname = ['a','b']
# columnname = ['z','x','c']
# df = pd.DataFrame(data=arr,index=rowname,columns=columnname)
# print(df)

url = "D:/workspace_py/pyproject/static/dataset"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv(url, names=names)
#print(data.describe)
array = data.values
X = array[:,0:8]
Y = array[:,8]

# num_trees = 200
# max_features = 5
# kfold = KFold(n_splits=10, random_state=7)
# model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
# results = cross_val_score(model, X, Y, cv=kfold)
# print(results.mean())

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on 33%
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)


# test = data.drop(['preg','plas','pres', 'test', 'mass', 'pedi', 'age'],axis=1);
# test.plot();
# plt.show();

#print(data.shape);
# plt.hist(data);
# #scatter_matrix(data)
# plt.show();

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()


# plt.plot([1,2,3,4], [1,4,9,16], 'yo')
# plt.axis([0, 6, 0, 20])
# plt.show()
#
# def f(t):
#     return np.exp(-t) * np.cos(2*np.pi*t)
# t1 = np.arange(0.0, 5.0, 0.1)
# t2 = np.arange(0.0, 5.0, 0.02)
# plt.figure(1)
# plt.subplot(221)
# plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
# plt.subplot(212)
# plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
# plt.show()
#
# mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)
#
# # the histogram of the data
# n, bins, patches = plt.hist(x,3, normed=1, facecolor='g', alpha=0.1)
#
#
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title('Histogram of IQ')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
# plt.grid(True)
# plt.show()


#test_id = data["preg"];
#print(data.drop(["preg"],axis=1))

# array = data.values
# #print(array);
# # separate array into input and output components
# X = array[:,0:8]
# #print(X);
# Y = array[:,8]
# #print(Y);
# # scaler = StandardScaler().fit(X)
# # rescaledX = scaler.transform(X)
# # # summarize transformed data
# # np.set_printoptions(precision=3)
# # print(rescaledX[0:5,:])
#
#
# kfold = KFold(n_splits=10, random_state=7)
# model = LogisticRegression()
# scoring = 'neg_log_loss'
# results = cross_val_score(model, X, Y, cv=kfold,scoring=scoring)
# print(results)
# print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)