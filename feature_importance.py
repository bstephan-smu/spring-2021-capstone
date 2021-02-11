#%%
from load_data import DataLoader
capData = DataLoader().load()
#pd.set_option('display.max_columns', None)


# %%

def get_data(self, data_cols='all', target_col='AD_event', alt_data=None):
    """
    \ndata_cols: pass the table prefix to return only a table from main options: 
    \n{'cpt_','lab_','vit_','medid_','asmt_icd', 'enc_'}
    \ntarget_col: pass AD_event or dem_event for target
    \nalt_data: pass in a list of columns to run grid on
    \ne.g. X, y = get_data(capData, data_cols='vit_')
    """
    df = self.main.copy()
    column_list = [target_col]# + alt_data

    if alt_data != None:
        column_list += alt_data

    dropcols = [col for col in df if col.lstrip('asmt_icd_') in self.dementia_icd_codes]
        
    response_cols = [
        'Cognition',
        'enc_id',
        'person_id',
        'AD_person',
        'AD_event',
        'dem_person',
        'dem_event',
        'ccsr_NVS011'
        ]
    
    response_cols.remove(target_col)
    
    dropcols += response_cols

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
    return (X,y)


def get_feature_importance(dataloader=capData, table='all', target='dem_person', alt_data=None):

    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score,\
        recall_score, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    #from sklearn.linear_model import LogisticRegression
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
        #clf = LogisticRegression()
        clf = RandomForestClassifier()
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

    # Get Feature Importances
    iDF = pd.DataFrame(zip(X,clf.feature_importances_), 
    columns=['Feature','Feature_Importance']).sort_values(
        by='Feature_Importance', ascending=False)

    print(iDF.nlargest(20, 'Feature_Importance'))
    
    return iDF


# %% less biased feature importance

# from sklearn.inspection import permutation_importance
# r = permutation_importance(clf, M, L,
#                            n_repeats=3,
#                            random_state=86)
# for i in r.importances_mean.argsort()[::-1]:
#     if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
#         print(f"{M.columns[i]:<20}"
#               f"{r.importances_mean[i]:.3f}"
#               f" +/- {r.importances_std[i]:.3f}")




# %% get all feature importance
FI_enc = get_feature_importance(capData, table='enc_')
FI_vit = get_feature_importance(capData, table='vit_')
FI_med = get_feature_importance(capData, table='med_')
FI_cpt = get_feature_importance(capData, table='cpt_')
FI_lab = get_feature_importance(capData, table='lab_')
FI_CCSR = get_feature_importance(capData, table='ccsr_')
FI_diag = get_feature_importance(capData, table='asmt_')
FI_all = get_feature_importance(capData)


# %%

import pandas as pd
cols = [FI_enc, FI_vit, FI_med, FI_diag, FI_cpt, FI_lab, FI_CCSR]
all_df = pd.DataFrame()
for df in cols:
    tab = df.Feature[0][:3]
    df.to_csv('FI_'+tab+'.csv')
    all_df = pd.concat([all_df, df[df['Feature_Importance'] > .001]])

FI_all = get_feature_importance(capData, alt_data=list(all_df.Feature), table='all', target='dem_person')
FI_all.to_csv('FI_all.csv')

FI_all.nlargest(30, 'Feature_Importance') 

# %%
FI_lab = get_feature_importance(capData, table='lab')
FI_lab

# %%
get_feature_importance(capData)

# %%
# %% GridSearch
from gridsearch import GridSearch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import warnings
from scipy.sparse.csc import csc_matrix
from sklearn.linear_model import SGDClassifier

warnings.simplefilter(action="default")

# Get Data:
data = get_data(capData, data_cols='xxx', target_col='dem_person', alt_data=list(LR_coefs['Feature']))

# Choose classifiers to run
classifiers = {
    'Random_Forest': RandomForestClassifier,
    'Logistic_Regression': LogisticRegression,
    'SVM': SVC,
    'GBoost': GradientBoostingClassifier,
    'AdaBoost': AdaBoostClassifier,
    'SGD': SGDClassifier
}

# Edit grid params.
# Hint: run RandomForestClassifier().get_params() to get param list.
param_grid = {
   'Random_Forest': {
    #     'bootstrap': True,
    #     'ccp_alpha': 0.0,
    #     'class_weight': ['balanced', None],
    #     'criterion': ['gini', 'entropy'],
    #     'max_depth': None,
    #    'max_features': ['auto'],
    #     'max_leaf_nodes': None,
    #     'max_samples': None,
    #     'min_impurity_decrease': 0.0,
    #     'min_impurity_split': None,
         'min_samples_leaf': [1,10,100],
         'min_samples_split': [2,30,500],
    #     'min_weight_fraction_leaf': 0.0,
         'n_estimators': [100, 1000]
    #     'n_jobs': None,
    #     'oob_score': False,
    #     'random_state': None,
    #     'verbose': 0,
    #     'warm_start': False
        },
    'Logistic_Regression': {
         'C': [.001,.01,.1,1,10,100], 
    #     'class_weight': ['balanced', None],
    #     # 'dual': False,
    #     # 'fit_intercept': True,
    #     # 'intercept_scaling': [1, 10],
    #     # 'l1_ratio': None,
         'max_iter': [1000],
         'multi_class': ['auto', 'ovr'],
    #     # 'n_jobs': None,
         'penalty': ['l2'], # sag requires L2 solver
    #     # 'random_state': None,
         'solver': ['sag'] # sag needs scaled data, but runs well on large datasets
    #     # 'tol': 0.0001,
    #     # 'verbose': 0,
    #     # 'warm_start': False
         },

    'SVM': {
         'C': [.001,.01,.1,1,10,100],
    #     'break_ties': False,
    #     'cache_size': 200,
    #     'class_weight': ['balanced', None],
    #     'coef0': 0.0,
    #     'decision_function_shape': 'ovr',
    #     'degree': 3,
    #     'gamma': ['scale', 'auto'],
         'kernel': ['linear', 'rbf']
    #     'max_iter': -1,
    #     'probability': False,
    #     'random_state': None,
    #     'shrinking': True,
    #     'tol': 0.001,
    #     'verbose': False
        },

    'GBoost' : {
        # 'ccp_alpha': 0.0,
        # 'criterion': 'friedman_mse',
        # 'init': None,
        # 'learning_rate': 0.1,
        # 'loss': 'deviance',
        # 'max_depth': 3,
        # 'max_features': None,
        # 'max_leaf_nodes': None,
        # 'min_impurity_decrease': 0.0,
        # 'min_impurity_split': None,
         'min_samples_leaf': [1,10,300],
         'min_samples_split': [3,30,100]
        # 'min_weight_fraction_leaf': 0.0,
        # 'n_estimators': 100,
        # 'n_iter_no_change': None,
        # 'presort': 'deprecated',
        # 'random_state': None,
        # 'subsample': 1.0,
        # 'tol': 0.0001,
        # 'validation_fraction': 0.1,
        # 'verbose': 0,
        # 'warm_start': False
        },

    'AdaBoost': {
        'algorithm': ['SAMME.R','SAMME'],
       # 'base_estimator': None,
        'learning_rate': [.1, 1, 10],
        'n_estimators': [50, 100, 1000],
       # 'random_state': None
        },

    'SGD': {
         'alpha':[.0001,.001,.01,.1,1],
        # 'average': False,
        # 'class_weight': None,
        # 'early_stopping': False,
        # 'epsilon': 0.1,
        # 'eta0': 0.0,
        # 'fit_intercept': True,
        # 'l1_ratio': 0.15,
        # 'learning_rate': 'optimal',
          'loss': ['squared_hinge', 'perceptron'],
          'max_iter': [5000], # Less than 1k will not resolve
        # 'n_iter_no_change': 5,
        # 'n_jobs': None,
          'penalty': ['l1','l2','elasticnet']
        # 'power_t': 0.5,
        # 'random_state': None,
        # 'shuffle': True,
        # 'tol': 0.001,
        # 'validation_fraction': 0.1,
        # 'verbose': 0,
        # 'warm_start': False
         },
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



# %%
# Get Logistic Regression Coefficients 
import pandas as pd
coefs = list(results.get('Logistic_Regression')['best_Logistic_Regression']['clf'].coef_[0])
labels = list(data[0])
LR_coefs = pd.DataFrame(zip(labels,coefs), columns = ['Feature', 'Coefficient']).sort_values(by='Coefficient', ascending=False)

LR_coefs[abs(LR_coefs.Coefficient) >  2]




# %% Get Random Forest Coefficients
coefs = results.get('Random_Forest')['best_Random_Forest']['clf'].feature_importances_
labels = list(data[0])

RF_coefs = pd.DataFrame(zip(labels,coefs), 
columns=['Feature','Feature_Importance']).sort_values(
    by='Feature_Importance', ascending=False)

RF_coefs[RF_coefs.Feature_Importance >  .001]


# %% Save Results Dict to pickle in your local dir

import pickle
with open('20210211results.pickle', 'wb') as picklefile:
    pickle.dump(results, picklefile)


# %% Print metrics per param setting:
for clf in results:
    print(clf)
    for x in results[clf]['iterations']:
        print(x['set_params'], '\nAUC: ', x['auc'],'\n F1: ', x['f1_score']'\nPrecision: ',x['precision'],'\nRecall: ',x['recall'],'\nAcc: ',x['accuracy'])