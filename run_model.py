#%%
from encoder import Encoder
capData = Encoder().load()
#pd.set_option('display.max_columns', None)


# Setup:
def get_data(self, data_cols='all', target_col='AD_encounter', alt_data=None, holdout=False):
    """
    \ndata_cols: pass the table prefix to return only a table from main options: 
    \n{'cpt_','lab_','vit_','medid_','asmt_icd', 'enc_'}
    \ntarget_col: pass AD_encounter or dem_encounter for target
    \nalt_data: pass in a list of columns to run grid on
    \ne.g. X, y = get_data(capData, data_cols='vit_')
    """
    from sklearn.model_selection import train_test_split

    df = self.main.copy()
    column_list = [target_col]# + alt_data

    if alt_data != None:
        column_list += alt_data
        
    response_cols = [
        'Cognition',
        'enc_id',
        'person_id',
        'AD_person',
        'AD_encounter',
        'dem_person',
        'dem_encounter',
        'ccsr_Neurocognitive disorders'
        ]
    
    response_cols.remove(target_col)
    
    dropcols = response_cols

    # Drop str cols
    dropcols += [col for col in capData.main if type(capData.main[col][0]) == str]

    # Drop list cols
    dropcols += [col for col in capData.main if type(capData.main[col][0]) == list]

    X = df.drop(columns=dropcols)
    
    if data_cols != 'all':
        column_list += [col for col in X if col.startswith(data_cols)]
        X = X[column_list]

    #X.dropna(inplace=True)
    X.fillna(0, inplace=True)
    X.reset_index(drop=True, inplace=True)
    y = X[target_col]
    X.drop(columns=target_col, inplace=True)

    if holdout:
        return train_test_split(X,y, test_size=.1, random_state=86, stratify=y)
    return (X,y)


def get_feature_importance(dataloader=capData, table='all', target='AD_person', alt_data=None):

    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score,\
        recall_score, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    #from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from scipy.sparse import csc_matrix

    X,y = get_data(dataloader, data_cols=table, target_col=target, alt_data=alt_data)
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(X, y)

    split_results = {}
    M = csc_matrix(X)
    L = y
    for split_id, (train_index, test_index) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(solver='sag',max_iter=5000, class_weight='balanced', C=.01)
        #clf = RandomForestClassifier()
        #from sklearn.ensemble import ExtraTreesClassifier
        #clf = ExtraTreesClassifier()
        try:  # always set random and all processors, w/o confounding param_grid
            clf.set_params(random_state=86)
            clf.set_params(n_jobs=-1)
        except:
            pass
        #clf.fit(M.iloc[train_index],L.iloc[train_index])
        clf.fit(M[train_index],L[train_index])

        #preds = clf.predict(M.iloc[test_index])
        preds = clf.predict(M[test_index])
        split_results[split_id] = {
            'train_index': train_index,
            'test_index': test_index,
            'split_accuracy': accuracy_score(L[test_index], preds),
            'split_precision': precision_score(L[test_index], preds),
            'split_recall': recall_score(L[test_index], preds),
            'split_f1_score': f1_score(L[test_index], preds),
            'split_auc': roc_auc_score(L[test_index], preds)
            }

    def split_avg(metric, split_results):
        metrics = [split_results[id][metric] for id in split_results]
        return np.mean(metrics)

    final_results = {
        'clf': clf,
        #'splits': split_results,
        'accuracy': split_avg('split_accuracy', split_results),
        'precision': split_avg('split_precision', split_results),
        'recall': split_avg('split_recall', split_results),
        'f1_score': split_avg('split_f1_score', split_results),
        'auc': split_avg('split_auc', split_results)
        }

    print(table, final_results)

    import pandas as pd
    coefs = list(final_results.get('clf').coef_[0])
    labels = list(X)

    LR_coefs = pd.DataFrame(zip(labels,coefs), columns = ['Feature', 'Coefficient']).sort_values(by='Coefficient', ascending=False)

    print(LR_coefs.nlargest(20, 'Coefficient'))
    
    return LR_coefs

# Run GridSearch
from gridsearch import GridSearch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
import warnings
from scipy.sparse.csc import csc_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
warnings.simplefilter(action="default")
import pandas as pd

try:
    FI_all = pd.read_csv('./ref/FI_all.csv')
except FileNotFoundError:
    FI_all = get_feature_importance(capData)

LR_coefs = pd.concat([FI_all.head(75) , FI_all.tail(75)])
#LR_coefs = LR_coefs[abs(LR_coefs.Coefficient) >  2]
LR_coefs = LR_coefs[LR_coefs['Feature'] != 'enc_Race_Native Hawaiian or Other Pacific Islander']
LR_coefs = LR_coefs[LR_coefs['Feature'] != 'enc_Race_ ']
LR_coefs = LR_coefs[LR_coefs['Feature'] != 'enc_AgeAtEnc']


# Get Data:
data = get_data(capData, data_cols='xxx', target_col='AD_person', alt_data=list(LR_coefs['Feature']))

# Choose classifiers to run
classifiers = {
    'Random_Forest': RandomForestClassifier,
    'Logistic_Regression': LogisticRegression,
    'XGBoost': XGBClassifier,
    'SVM': LinearSVC,
    #'GBoost': GradientBoostingClassifier,
    'AdaBoost': AdaBoostClassifier,
    'Naive_Bayes': BernoulliNB,
    'SGD': SGDClassifier
}

# Edit grid params.
# Hint: run RandomForestClassifier().get_params() to get param list.
param_grid = {
   'Random_Forest': {
    #     'bootstrap': True,
    #     'ccp_alpha': 0.0,
         'class_weight': [{1:6},{1:11}],
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': None,
    #    'max_features': ['auto'],
    #     'max_leaf_nodes': None,
    #     'max_samples': None,
    #     'min_impurity_decrease': 0.0,
    #     'min_impurity_split': None,
         'min_samples_leaf': [10,25,50],
         'min_samples_split': [25,75],
    #     'min_weight_fraction_leaf': 0.0,
         'n_estimators': [1000]
    #     'n_jobs': None,
    #     'oob_score': False,
    #     'random_state': None,
    #     'verbose': 0,
    #     'warm_start': False
        },
    'Logistic_Regression': {
         'C': [1,10], 
         'class_weight': [{1:6},'balanced'],
    #     # 'dual': False,
    #     # 'fit_intercept': True,
    #     # 'intercept_scaling': [1, 10],
    #     # 'l1_ratio': None,
         'max_iter': [5000],
         'multi_class': ['ovr'],
    #     # 'n_jobs': None,
         'penalty': ['l2'], # sag requires L2 solver
    #     # 'random_state': None,
         'solver': ['sag'], # sag needs scaled data, but runs well on large datasets
         'tol': [0.0001],
    #     # 'verbose': 0,
    #     # 'warm_start': False
         },

    'SVM' :{
        'C': [1],
        'class_weight': [{1:6},{1:11}], # 1/6 = dem_person, 1/11 = AD_person
        'dual': [False],
        # 'fit_intercept': True,
        # 'intercept_scaling': 1,
        'loss': ['squared_hinge'],
        'max_iter': [5000],
        # 'multi_class': 'ovr',
        'penalty': ['l2'],
        # 'random_state': None,
        # 'tol': 0.0001,
        # 'verbose': 0
         },

    'GBoost' : {
        'ccp_alpha': [0.01],
        # 'criterion': 'friedman_mse',
        # 'init': None,
        # 'learning_rate': [0.1, .01, .05],
        # 'loss': 'deviance',
         'max_depth': [20],
        # 'max_features': None,
        # 'max_leaf_nodes': None,
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
        # 'min_samples_leaf': [1,10,300],
        # 'min_samples_split': [3,30,100],
        # 'min_weight_fraction_leaf': 0.0,
         'n_estimators': [1000],
        # 'n_iter_no_change': None,
        # 'presort': 'deprecated',
        # 'random_state': None,
         'subsample': [.6],
         'tol': [0.0001, .001, .01],
        # 'validation_fraction': 0.1,
        # 'verbose': 0,
        # 'warm_start': False
        },

    'AdaBoost': {
        'algorithm': ['SAMME.R','SAMME'],
       # 'base_estimator': None,
        'learning_rate': [.1, .5, 1],
        'n_estimators': [1000],
       # 'random_state': None
        },

    'SGD': {
         'alpha':[.00001],
        # 'average': False,
         'class_weight': [{1:6},{1:11}],
        # 'early_stopping': False,
        # 'epsilon': 0.1,
        # 'eta0': 0.0,
        # 'fit_intercept': True,
        # 'l1_ratio': 0.15,
        # 'learning_rate': 'optimal',
          'loss': ['perceptron', 'squared_hinge'],
          'max_iter': [5000], # Less than 1k will not resolve
        # 'n_iter_no_change': 5,
        # 'n_jobs': None,
          'penalty': ['l2']
        # 'power_t': 0.5,
        # 'random_state': None,
        # 'shuffle': True,
        # 'tol': 0.001,
        # 'validation_fraction': 0.1,
        # 'verbose': 0,
        # 'warm_start': False
         },

    'Naive_Bayes' : {
        'alpha':[.01,.1,1],
        'binarize':[.1],
        'fit_prior':[True,False],
        # 'class_prior':
    },

    'XGBoost' : {
     'objective': ['binary:logistic'],
    # 'use_label_encoder': True,
    # 'base_score': None,
     'booster': ['gbtree'],
     'colsample_bylevel': [1],
    # 'colsample_bynode': None,
     'colsample_bytree': [.8],
     'gamma': [0],
    # 'gpu_id': None,
    # 'importance_type': 'gain',
    # 'interaction_constraints': None,
    # 'learning_rate': None,
     'max_delta_step': [0],
     'max_depth': [10, 20],
     'min_child_weight': [15],
    # 'missing': nan,
    # 'monotone_constraints': None,
    # 'n_estimators': 100,
    # 'n_jobs': None,
    # 'num_parallel_tree': None,
    # 'random_state': None,
     'reg_alpha': [.01, .025],
     'reg_lambda': [.01, .025],
    # 'scale_pos_weight': None,
     'subsample': [.6],
     'tree_method': ['hist']
    # 'validate_parameters': None,
    # 'verbosity': None
    }
}

gs = GridSearch()
gs.set_params(
    data=data,
    classifiers=classifiers,
    param_grid=param_grid,
    metric='auc'
)
results = gs.run_grid(n_folds=3, splits=False, scale=True, sparse=True, verbose=True)
gs.plot_metrics(save=True)



# Get Logistic Regression Coefficients 
import pandas as pd
coefs = list(results.get('Logistic_Regression')['best_Logistic_Regression']['clf'].coef_[0])
labels = list(data[0])
best_LR_coefs = pd.DataFrame(zip(labels,coefs), columns = ['Feature', 'Coefficient']).sort_values(by='Coefficient', ascending=False)
print('Logistic Regression Coefs > abs(1)')
pd.display(best_LR_coefs[abs(best_LR_coefs.Coefficient) >  1])


#  Get Random Forest Coefficients
coefs = results.get('Random_Forest')['best_Random_Forest']['clf'].feature_importances_
labels = list(data[0])
RF_coefs = pd.DataFrame(zip(labels,coefs), 
columns=['Feature','Feature_Importance']).sort_values(
    by='Feature_Importance', ascending=False)
print('Random Forest Feature Importances > .001')
pd.display(RF_coefs[RF_coefs.Feature_Importance >  .001])


# Save Results Dict to pickle in your local dir
import pickle
with open('./GridSearch/results.pickle', 'wb') as picklefile:
    pickle.dump(results, picklefile)

# %% Run model on holdout set
from sklearn.metrics import accuracy_score, f1_score, precision_score,\
    recall_score, roc_auc_score, confusion_matrix

X_train, X_test, y_train, y_test = get_data(capData, data_cols='xxx', target_col='AD_person', alt_data=list(LR_coefs['Feature']), holdout=True)
clf = results['best_overall']['clf']
clf.fit(X_train,X_test)
preds = clf.predict(y_train)

holdout_results = {
    'accuracy': accuracy_score(
        y_test, preds),
    'precision': precision_score(
        y_test, preds),
    'recall': recall_score(
        y_test, preds),
    'f1_score': f1_score(
        y_test, preds),
    'auc': roc_auc_score(
        y_test, preds),
    'specificity': gs.specificity_score(
        y_test, preds),
    'ppv': gs.ppv_score(
        y_test, preds),
    'npv': gs.npv_score(
        y_test, preds),                            
    }

gs.results = holdout_results
gs.plot_metrics()
print(holdout_results)