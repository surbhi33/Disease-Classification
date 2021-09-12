


import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

import gc
gc.collect()

################BASIC STATS##############
combined_data=pd.read_excel('C:/Users/_/Desktop/Combined two graphs _ data v1.xlsx',sheet_name='Final Data')
combined_data.info()
combined_data.describe()
len(combined_data.index)
cort=combined_data.corr()
print(cort)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(cort, vmax=.8, square=True)
combined_data['Score'].std()
combined_data['Score'].skew()
combined_data['Score'].kurt()

combined_data['log_Age']=np.log(combined_data['Age'])

######SUMMARY PLOTS############
sns.set(style="whitegrid")
plt.figure(figsize=(10,8))
ax = sns.boxplot(x='Score', data=combined_data, orient="v")
print(ax)
ax1 = sns.boxplot(x='Stage', y='Score', data=combined_data, orient="v")
print(ax1)

var = 'Age'
data = pd.concat([combined_data['Score'], combined_data[var]], axis=1)
fig=data.plot.scatter(x=var, y='Score', ylim=(0,30))

cat=combined_data['Stage'].value_counts()
print(cat)


############SPLIT#############


X=combined_data
X1=X.columns.drop(['Score','Clubbed Stage','Stage','Age_mod'])
print(X1)
y=combined_data['Stage']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X[X1], y, test_size = 0.50, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

import numpy as np

X_train.info()
np.dtype(y_train)

###############SMOTE ON TRAIN DATA################

X_train['Stage'].value_counts().plot(kind='bar', title='Count (Stage')

from imblearn.over_sampling import SMOTE

smote = SMOTE( n_jobs=-1)
smote_min = SMOTE( sampling_strategy='minority',n_jobs=-1)
X_sm_all, y_sm_all = smote.fit_resample(X[X1], y)
X_sm, y_sm = smote.fit_resample(X_train, y_train)
X_sm_min, y_sm_min = smote_min.fit_resample(X_train, y_train)
print(X_sm)
print(y_sm)

##PLOT SMOTE DATA###
df = pd.DataFrame(X_sm, columns=X1)
df['Stage'] = y_sm

cat_df=df['Stage'].value_counts()
print(cat_df)

df['Stage'].value_counts().plot(kind='bar', title='Count (Stage)')

##################LOGISTIC REGRESSION####################

############K FOLD VALIDATION#####################

CV_model = cross_val_predict(LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2',C=0.1, solver='newton-cg'), X_sm_all, y_sm_all, cv=10)
print(CV_model)
print('Accuracy Score:', metrics.accuracy_score(y_sm_all, CV_model)) 


#######################################################

model2 = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2',C=0.1, solver='newton-cg').fit(X_sm_min, y_sm_min)
preds = model2.predict(X_test)
print(preds)

#GET PARAM#####
params = model2.get_params()
print(params)

##GET COEFF##############

print('Intercept: \n', model2.intercept_)
print('Coefficients: \n', model2.coef_)
coeff_df=pd.DataFrame(model2.intercept_)

#####GET CM############
cm=np.array(confusion_matrix(y_test, preds))
print(cm)

cm_df=pd.DataFrame(cm, index=['Original_1','Original_2', 'Original_3'],columns=['predicted_1', 'predicted_2', 'predicted_3'])


#####ACCURACY###########
print('Accuracy Score:', metrics.accuracy_score(y_test, preds)) 
test_acc= metrics.accuracy_score(y_test, preds)
print('Accuracy Score:', metrics.accuracy_score(y_train, model2.predict(X_train)))
train_acc=metrics.accuracy_score(y_train, model2.predict(X_train))
print('Accuracy Score:', metrics.accuracy_score(combined_data['Clubbed Stage'], model2.predict(combined_data[X1])))  
all_data_acc=metrics.accuracy_score(combined_data['Clubbed Stage'], model2.predict(combined_data[X1]))

test_acc_df=pd.DataFrame(test_acc,index=['Cohort'],columns=['Test Accuracy'])
print(test_acc_df)
test_acc_df['Train Accuracy'] = train_acc
print(test_acc_df)
test_acc_df['All Data Accuracy'] = all_data_acc
print(test_acc_df)

#######GET F1#######
class_report=classification_report(y_test, preds,output_dict=True)
print(class_report)

report_df=pd.DataFrame(class_report).transpose()
print(report_df)



####GET PROPENSITY SCORES#############
propensity=model2.predict_proba(X_test)
propensity_df=pd.DataFrame(propensity)

#############EXPORT RESULTS##########
writer = pd.ExcelWriter('C:/Users/_/Desktop/final_results_model1.xlsx')
cm_df.to_excel(writer,'Confusion matrix')
propensity_df.to_excel(writer,'Propensity Score')
coeff_df.to_excel(writer,'Regression coeffecients')
test_acc_df.to_excel(writer,'Accuracy')
report_df.to_excel(writer,'class report')
writer.save()

################3
##############SVM ##################
clf_svm = svm.SVC(kernel='rbf', C=10).fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svm))
clf_svm.score(X_test, y_test)

cm_5=confusion_matrix(y_test, y_pred_svm)
print(cm_5)
print('Accuracy Score:', metrics.accuracy_score(y_test, y_pred_svm))
print('Accuracy Score:', metrics.accuracy_score(y_train, clf_svm.predict(X_train)))
print('Accuracy Score:', metrics.accuracy_score(y_train, clf_svm.predict(X_train)))

class_report=classification_report(y_test, y_pred_svm)
print(class_report)


clf_svm = svm.SVC(kernel='rbf', C=100).fit(X_sm, y_sm)
y_pred_svm = clf_svm.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_svm))
print('Accuracy Score:', metrics.accuracy_score(y_sm, clf_svm.predict(X_sm)))
clf_svm.score(X_test, y_test)

cm_5=confusion_matrix(y_test, y_pred_svm)
print(cm_5)



##############XGBMclassifier#################
from xgboost import XGBClassifier

m = XGBClassifier(
    max_depth=2,
    gamma=2,
    eta=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5
)

model3=m.fit(X_sm, y_sm)
m.predict_proba(X_test)[:,1]

preds_3 = model2.predict(X_test)


cm=confusion_matrix(y_test, preds_3)
print(cm)
print('Accuracy Score:', metrics.accuracy_score(y_test, preds_3))
print('Accuracy Score:', metrics.accuracy_score(y_sm, model3.predict(X_sm)))



###########CART/Decision Stump####################
clf_dt = DecisionTreeClassifier(criterion="entropy", max_depth=5)
clf_dt = clf_dt.fit(X_sm,y_sm)
y_pred = clf_dt.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
cm_4=confusion_matrix(y_test, y_pred)
print(cm_4)


############
 rf = RandomForestClassifier(n_estimators = 10,max_depth=3,min_samples_split=2)
rf.fit(X_sm,y_sm)
y_pred_rf= clf_dt.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rf))


######



model = DecisionTreeClassifier(random_state=0)
params = {
    'criterion': ['entropy', 'gini'],
    'max_features': [None, 'auto'],
    'min_samples_split': [i / 10.0 for i in range(1, 10)],
    'min_samples_leaf': [i / 10.0 for i in range(1, 5)]
}

print(params)
  n_splits = 2
  n_repeats = 5
  cv = model_selection.RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,random_state=2652124)
  clf_gs = model_selection.GridSearchCV(model, params, cv=cv, scoring='accuracy')
  clf_gs.fit(X_sm, y_sm)
  
   scoring = {
        'accuracy':  metrics.make_scorer(metrics.accuracy_score),
        'f1':        metrics.make_scorer(metrics.f1_score),
        'precision': metrics.make_scorer(metrics.precision_score),
        'recall':    metrics.make_scorer(metrics.recall_score)
    }
   
    validation = model_selection.cross_validate(clf_gs.best_estimator_, X_test, y_test, scoring=scoring, cv=cv)
    
    scores = { score: [np.mean(validation['test_{}'.format(score)])] for score in scoring.keys()
