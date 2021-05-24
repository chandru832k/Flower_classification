import pandas as pd
import pickle
from sklearn import preprocessing, metrics, tree
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from matplotlib import pyplot as plt
import numpy as np
from sklearn.externals import joblib

train=pd.read_csv("data/iris.csv")
train.head()
train.describe()
train.describe(include="O")
for column in train.columns:
    train[column].fillna(train[column].mode()[0],inplace=True)
    cols=list(train.columns)
X_train=train.loc[:,cols[0:-1]]
y_train=train.loc[:,cols[-1]]

test=pd.read_csv("data/iris.csv")
test.head()
X_test=test.loc[:,cols[0:-1]]
y_test=test.loc[:,cols[-1]]

model_lr = LogisticRegression(random_state = 0)
model_lr.fit(X_train, y_train)
joblib.dump(model_lr, 'logit_model_iris.pkl')
y_pred = model_lr.predict(X_test)
lr_acc = accuracy_score(y_test, y_pred)
lr_ps = precision_score(y_test, y_pred, average='macro', zero_division= 0 )
lr_f1 = f1_score(y_test, y_pred, average='macro', zero_division= 0 )
lr_rs = recall_score(y_test, y_pred, average='macro', zero_division= 0)
#print(classification_report(y_test, y_pred, zero_division= 0 ))

model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
joblib.dump(model_nb, 'nb_model_iris.pkl')
y_pred = model_nb.predict(X_test)
nb_acc = accuracy_score(y_test, y_pred)
nb_ps = precision_score(y_test, y_pred, average='macro', zero_division= 0 )
nb_f1 = f1_score(y_test, y_pred, average='macro', zero_division= 0 )
nb_rs = recall_score(y_test, y_pred, average='macro', zero_division= 0)
#print(classification_report(y_test, y_pred, zero_division= 0 ))

model_dt = tree.DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
joblib.dump(model_dt, 'dtree_model_iris.pkl')
y_pred = model_dt.predict(X_test)
dt_acc = accuracy_score(y_test, y_pred)
dt_f1 = f1_score(y_test, y_pred, average='macro', zero_division= 0 )
dt_ps = precision_score(y_test, y_pred, average='macro', zero_division= 0 )
dt_rs = recall_score(y_test, y_pred, average='macro', zero_division= 0)
#print(classification_report(y_test, y_pred, zero_division= 0 ))

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)
joblib.dump(model_knn, 'knn_model_iris.pkl')
y_pred = model_knn.predict(X_test)
knn_acc = accuracy_score(y_test, y_pred)
knn_f1 = f1_score(y_test, y_pred, average='macro', zero_division= 0 )
knn_ps = precision_score(y_test, y_pred, average='macro', zero_division= 0 )
knn_rs = recall_score(y_test, y_pred, average='macro', zero_division= 0)
#print(classification_report(y_test, y_pred, zero_division= 0 ))

model_lda = LDA()
model_lda.fit(X_train, y_train)
joblib.dump(model_lda, 'lda_model_iris.pkl')
y_pred = model_lda.predict(X_test)
lda_acc = accuracy_score(y_test, y_pred)
lda_f1 = f1_score(y_test, y_pred, average='macro', zero_division= 0 )
lda_ps = precision_score(y_test, y_pred, average='macro', zero_division= 0 )
lda_rs = recall_score(y_test, y_pred, average='macro', zero_division= 0)
#print(classification_report(y_test, y_pred, zero_division= 0 ))

model_svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
model_svm.fit(X_train, y_train)
joblib.dump(model_svm, 'svm_model_iris.pkl')
y_pred = model_svm.predict(X_test)
svm_acc = metrics.accuracy_score(y_test, y_pred)
svm_f1 = metrics.f1_score(y_test, y_pred, average='macro', zero_division= 0 )
svm_ps = metrics.precision_score(y_test, y_pred, average='macro', zero_division= 0 )
svm_rs =  metrics.recall_score(y_test, y_pred, average='macro', zero_division= 0)

def round_off(str, acc, f1, ps, rs):
  print(str, round(acc,2), round(f1,2), round(ps,2), round(rs,2))

print("     Acc  F1   Pre  Rec")
round_off("KNN", knn_acc, knn_f1, knn_ps, knn_rs)
round_off("DT ", dt_acc, dt_f1, dt_ps, dt_rs)
round_off("NB ", nb_acc, nb_f1, nb_ps, nb_rs)
round_off("SVM", svm_acc, svm_f1, svm_ps, svm_rs)
round_off("LDA", lda_acc, lda_f1, lda_ps, lda_rs)
round_off("LR ", lr_acc, lr_f1, lr_ps, lr_rs)

#print (metrics.classification_report(y_test, y_pred, zero_division= 0 ))

def plot(score, col, y_lab, tit):
  fig = plt.figure(figsize =(8, 6))
  algo = ['KNN', 'NB', 'Dtree', 'SVM', 'LDA', 'LR']
  plt.bar(algo,score,color = col, width = 0.4)
  plt.xlabel('Classification Algorithms')
  plt.ylabel(y_lab)
  plt.title(tit)
  plt.show()

acc = [knn_acc, nb_acc, dt_acc, svm_acc, lda_acc, lr_acc]
plot(acc, "orange", "Accuracy", "Accuracy comparision")

ps = [knn_ps, nb_ps, dt_ps, svm_ps, lda_ps, lr_ps]
plot(acc, "violet", "Precision Score", "Precision Score comparision")

f1 = [knn_f1, nb_f1, dt_f1, svm_f1, lda_f1, lr_f1]
plot(acc, "red", "F1 score", "F1 score comparision")

rs = [knn_rs, nb_rs, dt_rs, svm_rs, lda_rs, lr_rs]
plot(acc, "blue", "Recall score", "Recall score comparision")