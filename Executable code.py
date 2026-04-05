'# Machine Learning Models and Training 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
data0 = pd.read_csv('/content/5.urldata.csv') 
data0.head() 
data0.shape 
data0.columns 
data0.info() 
data0.describe() 
data = data0.drop(['Domain'], axis = 1).copy() 
data.isnull().sum() 
data = data.sample(frac=1).reset_index(drop=True) 
data.head() 
y = data['Label'] 
X = data.drop('Label',axis=1) 
X.shape, y.shape 
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 12) 
X_train.shape, X_test.shape 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score 
ML_Model = [] 
acc_train = [] 
acc_test = [] 
prec_train=[] 
prec_test=[] 
recall_train=[] 
recall_test=[] 
f1_train=[] 
f1_test=[] 
#function to call for storing the results 
def storeResults(model, a,b,c,d,e,f,g,h): 
ML_Model.append(model) 
acc_train.append(round(a, 9)) 
acc_test.append(round(b, 9)) 
prec_train.append(round(c, 9)) 
prec_test.append(round(d, 9)) 
recall_train.append(round(e, 9)) 
recall_test.append(round(f, 9)) 
f1_train.append(round(g, 9)) 
36 
f1_test.append(round(h, 9)) 
#DECISION TREE 
from sklearn.tree import DecisionTreeClassifier 
tree = DecisionTreeClassifier(max_depth = 5) # instantiate the model 
tree.fit(X_train, y_train) # fit the model 
y_test_tree = tree.predict(X_test) 
y_train_tree = tree.predict(X_train) 
acc_train_tree = accuracy_score(y_train,y_train_tree) 
acc_test_tree = accuracy_score(y_test,y_test_tree) 
prec_train_tree = precision_score(y_train,y_train_tree) 
prec_test_tree = precision_score(y_test,y_test_tree) 
recall_train_tree = recall_score(y_train,y_train_tree) 
recall_test_tree = recall_score(y_test,y_test_tree) 
f1_train_tree = f1_score(y_train,y_train_tree) 
f1_test_tree = f1_score(y_test,y_test_tree) 
print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree)) 
print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree)) 
print("Decision Tree: precision on training Data: {:.3f}".format(prec_train_tree)) 
print("Decision Tree: precision on test Data: {:.3f}".format(prec_test_tree)) 
print("Decision Tree: recall on training Data: {:.3f}".format(recall_train_tree)) 
print("Decision Tree: recall on test Data: {:.3f}".format(recall_test_tree)) 
print("Decision Tree: f1score on training Data: {:.3f}".format(f1_train_tree)) 
print("Decision Tree: f1score on test Data: {:.3f}".format(f1_test_tree)) 
storeResults('DecisionTree',acc_train_tree,acc_test_tree,prec_train_tree,prec_test_tree,recall_train_tr 
ee,recall_test_tree,f1_train_tree,f1_test_tree) 
#RANDOM FOREST 
from sklearn.ensemble import RandomForestClassifier 
forest = RandomForestClassifier(max_depth=5) # instantiate the model 
forest.fit(X_train, y_train) # fit the model 
y_test_forest = forest.predict(X_test) 
y_train_forest = forest.predict(X_train) 
acc_train_forest = accuracy_score(y_train,y_train_forest) 
acc_test_forest = accuracy_score(y_test,y_test_forest) 
prec_train_forest= precision_score(y_train,y_train_forest) 
prec_test_forest = precision_score(y_test,y_test_forest) 
recall_train_forest = recall_score(y_train,y_train_forest) 
recall_test_forest = recall_score(y_test,y_test_forest) 
f1_train_forest = f1_score(y_train,y_train_forest) 
f1_test_forest = f1_score(y_test,y_test_forest) 
print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest)) 
print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest)) 
37 
print("Random forest:precision on training Data: {:.3f}".format(prec_train_forest)) 
print("Random forest: precision on test Data: {:.3f}".format(prec_test_forest)) 
print("Random forest: recall on training Data: {:.3f}".format(recall_train_forest)) 
print("Random forest: recall on test Data: {:.3f}".format(recall_test_forest)) 
print("Random forest: f1 score on training Data: {:.3f}".format(f1_train_forest)) 
print("Random forest: f1 score on test Data: {:.3f}".format(f1_test_forest)) 
storeResults('randomforest',acc_train_forest,acc_test_forest,prec_train_forest,prec_test_forest,recall_ 
train_forest,recall_test_forest,f1_train_forest,f1_test_forest) 
#XGBOOST 
from xgboost import XGBClassifier 
xgb = XGBClassifier(learning_rate=0.4,max_depth=7) # instantiate the model 
xgb.fit(X_train, y_train) #fit the model 
y_test_xgb = xgb.predict(X_test) 
y_train_xgb = xgb.predict(X_train) 
acc_train_xgb = accuracy_score(y_train,y_train_xgb) 
acc_test_xgb = accuracy_score(y_test,y_test_xgb) 
prec_train_xgb= precision_score(y_train,y_train_xgb) 
prec_test_xgb = precision_score(y_test,y_test_xgb) 
recall_train_xgb = recall_score(y_train,y_train_xgb) 
recall_test_xgb = recall_score(y_test,y_test_xgb) 
f1_train_xgb = f1_score(y_train,y_train_xgb) 
f1_test_xgb = f1_score(y_test,y_test_xgb) 
print("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_xgb)) 
print("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_xgb)) 
print(" xgb:precision on training Data: {:.3f}".format(prec_train_xgb)) 
print(" xgb: precision on test Data: {:.3f}".format(prec_test_xgb)) 
print(" xgb: recall on training Data: {:.3f}".format(recall_train_xgb)) 
print(" xgb: recall on test Data: {:.3f}".format(recall_test_xgb)) 
print(" xgb: f1 score on training Data: {:.3f}".format(f1_train_xgb)) 
print(" xgb: f1 score on test Data: {:.3f}".format(f1_test_xgb)) 
storeResults('xgb',acc_train_xgb,acc_test_xgb,prec_train_xgb,prec_test_xgb,recall_train_xgb,recall_t 
est_xgb,f1_train_xgb,f1_test_xgb) 
#SUPPORT VECTOR MACHINE(SVM) 
from sklearn.svm import SVC 
svm = SVC(kernel='linear', C=1.0, random_state=12) # instantiate the model 
svm.fit(X_train, y_train)#fit the model 
y_test_svm = svm.predict(X_test) 
y_train_svm = svm.predict(X_train) 
acc_train_svm = accuracy_score(y_train,y_train_svm) 
38 
acc_test_svm = accuracy_score(y_test,y_test_svm) 
prec_train_svm= precision_score(y_train,y_train_svm) 
prec_test_svm = precision_score(y_test,y_test_svm) 
recall_train_svm = recall_score(y_train,y_train_svm) 
recall_test_svm = recall_score(y_test,y_test_svm) 
f1_train_svm = f1_score(y_train,y_train_svm) 
f1_test_svm = f1_score(y_test,y_test_svm) 
print("SVM: Accuracy on training Data: {:.3f}".format(acc_train_svm)) 
print("SVM : Accuracy on test Data: {:.3f}".format(acc_test_svm)) 
print(" svm:precision on training Data: {:.3f}".format(prec_train_svm)) 
print(" svm: precision on test Data: {:.3f}".format(prec_test_svm)) 
print(" svm: recall on training Data: {:.3f}".format(recall_train_svm)) 
print(" svm: recall on test Data: {:.3f}".format(recall_test_svm)) 
print(" svm: f1 score on training Data: {:.3f}".format(f1_train_svm)) 
print(" svm: f1 score on test Data: {:.3f}".format(f1_test_svm)) 
storeResults('svm',acc_train_svm,acc_test_svm,prec_train_svm,prec_test_svm,recall_train_svm,recal 
l_test_svm,f1_train_svm,f1_test_svm) 
#K NEAREST NEIGHBORS(KNN) 
from sklearn.neighbors import KNeighborsClassifier 
knn= KNeighborsClassifier(n_neighbors=5) 
knn.fit(X_train, y_train) 
y_test_knn = knn.predict(X_test) 
y_train_knn = knn.predict(X_train) 
acc_train_knn = accuracy_score(y_train,y_train_knn) 
acc_test_knn = accuracy_score(y_test,y_test_knn) 
prec_train_knn= precision_score(y_train,y_train_knn) 
prec_test_knn = precision_score(y_test,y_test_knn) 
recall_train_knn = recall_score(y_train,y_train_knn) 
recall_test_knn = recall_score(y_test,y_test_knn) 
f1_train_knn = f1_score(y_train,y_train_knn) 
f1_test_knn = f1_score(y_test,y_test_knn) 
print("knn: Accuracy on training Data: {:.3f}".format(acc_train_knn)) 
print("knn: Accuracy on test Data: {:.3f}".format(acc_test_knn)) 
print(" knn:precision on training Data: {:.3f}".format(prec_train_knn)) 
print(" knn: precision on test Data: {:.3f}".format(prec_test_knn)) 
print(" knn: recall on training Data: {:.3f}".format(recall_train_knn)) 
print(" knn: recall on test Data: {:.3f}".format(recall_test_knn)) 
print(" knn: f1 score on training Data: {:.3f}".format(f1_train_knn)) 
print(" knn: f1 score on test Data: {:.3f}".format(f1_test_knn)) 
storeResults('knn',acc_train_knn,acc_test_knn,prec_train_knn,prec_test_knn,recall_train_knn,recall_t 
est_knn,f1_train_knn,f1_test_knn) 
results = pd.DataFrame({ 'ML Model': ML_Model, 
'Train Accuracy': acc_train, 
39 
'Test Accuracy': acc_test, 
'Train Precision': prec_train, 
‘Test Precision': prec_test, 
'Train recall': recall_train, 
'Test recall': recall_test, 
'Train f1 score': f1_train, 
'Test f1 score': f1_test}) 
results 
results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False) 
import matplotlib.pyplot as plt 
import pandas as pd 
# Create a pandas dataframe from the data 
data = { 
'ML Model': ['Decision Tree', 'Random Forest', 'XGB', 'SVM', 'KNN'], 
'Train Accuracy': [0.811250, 0.814625, 0.866125, 0.799375, 0.852875], 
'Test Accuracy': [0.8215, 0.8225, 0.8665, 0.8130, 0.8605], 
'Train Precision': [0.975858, 0.980769, 0.921697, 0.967643, 0.932990], 
'Test Precision': [0.967791, 0.978125, 0.902299, 0.966929, 0.925743], 
'Train recall': [0.641458, 0.644929, 0.802628, 0.622861, 0.762956], 
'Test recall': [0.652534, 0.647363, 0.811789, 0.634953, 0.773526], 
'Train f1 score': [0.774087, 0.778160, 0.858052, 0.757882, 0.839449], 
'Test f1 score': [0.779494, 0.779091, 0.854654, 0.766542, 0.842817] 
} 
df = pd.DataFrame(data) 
# Plot the graph 
fig, ax = plt.subplots(figsize=(10, 6)) 
ax.plot(df['ML Model'], df['Train Accuracy'], label='Train Accuracy') 
ax.plot(df['ML Model'], df['Test Accuracy'], label='Test Accuracy') 
ax.plot(df['ML Model'], df['Train Precision'], label='Train Precision') 
ax.plot(df['ML Model'], df['Test Precision'], label='Test Precision') 
ax.plot(df['ML Model'], df['Train recall'], label='Train recall') 
ax.plot(df['ML Model'], df['Test recall'], label='Test recall') 
ax.plot(df['ML Model'], df['Train f1 score'], label='Train f1 score') 
ax.plot(df['ML Model'], df['Test f1 score'], label='Test f1 score') 
ax.set_xlabel('ML Model') 
ax.set_ylabel('Score') 
ax.set_title('ML Model Performance Metrics') 
ax.legend() 
plt.show() `
