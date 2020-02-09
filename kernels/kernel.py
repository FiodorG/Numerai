import numpy as np
from libraries import utilities
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import *
from sklearn.metrics import log_loss, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


#########################################################
def train_linear_models(x_train, y_train, x_validation, y_validation, x_test, verbose=False):
    
    alphas = list(np.power(10.0, np.arange(-10, 10)))
    cv = StratifiedKFold(n_splits=4)
    
    scores = {}
    models = {}
    predictions = {}
    
    for i in np.arange(x_train.shape[1]):
        model = LogisticRegressionCV(Cs=alphas, cv=cv, penalty='l2', n_jobs=-1)
        model.fit(x_train[:, i:i+1], y_train)
        y_predicted = model.predict_proba(x_validation[:, i:i+1])
        score = log_loss(y_validation, y_predicted)
        
        if score < -log(0.5):
            scores[columns[i]] = score 
            models[columns[i]] = model
            predictions[columns[i]] = model.predict_proba(x_test[:, i:i+1])[:, 1]
                  
        if verbose:
            print(sklearn.metrics.classification_report(y_train, y_predicted))
            confusion_matrix(y_train, y_predicted)        
    
    return scores, models, predictions


#########################################################
df_train, df_test = utilities.get_data(n_first_train=-1)
columns = utilities.get_columns(df_train, with_target=False)

x_train = df_train.drop('target', axis=1)
y_train = df_train[['target']].values
x_test = df_test.drop('t_id', axis=1)

scaler = preprocessing.StandardScaler()
scaler = preprocessing.Normalizer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


#################### Model1
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
scores, models, predictions = train_linear_models(x_train, y_train, x_validation, y_validation, x_test)

y_predicted_linear = np.median(list(predictions.values()), axis=0)


#################### Model4
rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=1, n_jobs=-1)
clf1 = KNeighborsClassifier(n_neighbors=100)
clf2 = LogisticRegression(penalty='l1', C=0.1)
clf3 = GaussianNB()
sclf = StackingClassifier(classifiers=[clf1,clf2,clf3], meta_classifier=rf, use_probas=True, average_probas=False)
cross_val_score(sclf, x_train, y_train, cv=3)
sclf.fit(x_train, y_train)
results = svm.predict_proba(x_test)[:, 1]


#################### Model2
n_components = utilities.pca_get_components(np.append(x_train, x_test, axis=0), min_explained_variance=0.98)
pca = PCA(n_components=n_components)
pca.fit(np.append(x_train, x_test, axis=0))
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

poly = PolynomialFeatures(2)
x_train = poly.fit_transform(x_train)
x_test = poly.fit_transform(x_test)

cv = KFold(n_splits=4)
tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100]},
#        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
]

scores = ['precision']
# , 'recall'

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=cv, scoring='%s_macro' % score, n_jobs=-1)
    clf.fit(x_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

svm = SVC(kernel='rbf', C=10, probability=True)
cross_val_score(svm, x_train, y_train, cv=cv)
svm.fit(x_train, y_train)
results = svm.predict_proba(x_test)[:, 1]


#################### Model3
xgmat_train = xgb.DMatrix(x_train, label=y_train)
xgmat_valid = xgb.DMatrix(x_validation, label=y_validation)
xgmat_test = xgb.DMatrix(x_test)

params_xgb = {
'booster': 'dart',
'objective': 'binary:logistic',
'nthread ': 12,

'max_depth': 3,
'learning_rate': 0.10,
'subsample': 0.5,
'base_score': 0.5,

'reg_alpha': 0.01,
'reg_lambda': 1,

'rate_drop': 0.1,
'skip_drop': 0.5,
'normalize_type': 'forest',
'sample_type': 'weighted',

'eval_metric': 'error'
}

num_rounds = 2000
bst = xgb.train(params_xgb, xgmat_train, num_rounds)
params_xgb.update({'process_type': 'update', 'updater': 'refresh', 'refresh_leaf': False})
bst_after = xgb.train(params_xgb, xgmat_valid, num_rounds, xgb_model=bst)

imp = pd.DataFrame()
imp['train'] = pd.Series(bst.get_score(importance_type='gain'))
imp['OOB'] = pd.Series(bst_after.get_score(importance_type='gain'))
ax = imp.sort_values('OOB').tail(50).plot.barh(title='Feature importances sorted by OOB', figsize=(7,4))

y_train_predicted = bst_after.predict(xgmat_train, ntree_limit=num_rounds)
y_validation_predicted = bst_after.predict(xgmat_valid, ntree_limit=num_rounds)

print("Logloss Train: %.2f%%" % (log_loss(y_train, y_train_predicted) * 100.0))
print("Logloss Valid: %.2f%%" % (log_loss(y_validation, y_validation_predicted) * 100.0))

y_train_predicted = [round(value) for value in y_train_predicted]
y_validation_predicted = [round(value) for value in y_validation_predicted]

print("Accuracy Train: %.2f%%" % (accuracy_score(y_train, y_train_predicted) * 100.0))
print("Accuracy Valid: %.2f%%" % (accuracy_score(y_validation, y_validation_predicted) * 100.0))

y_predicted_xgb = bst_after.predict(xgmat_test, ntree_limit=num_rounds)

y_predicted = 0.5 * y_predicted_xgb + 0.5 * y_predicted_linear

utilities.save_results(df_test, y_predicted)
