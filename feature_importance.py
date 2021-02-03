#%%
from load_data import DataLoader
capData = DataLoader()
#capData.create(name='clean')
capData = capData.load('clean')

# %%
def get_data(self, data_cols='all', target_col='AD_event'):
    """
    \ndata_cols: pass the table prefix to return only a table from main
    \ntarget_col: pass AD_event or Dem_event for target
    \ne.g. X, y = get_data(capData, data_cols='vit_')
    """
    df = self.main.copy()
    dropcols = [col for col in df if col.lstrip('asmt_icd_') in self.dementia_icd_codes]
    X = df.drop(columns=dropcols)

    if data_cols != 'all':
        column_list = ['enc_id','person_id', target_col]
        column_list += [col for col in X if col.startswith(data_cols)]
        X = X[column_list]
    X.dropna(inplace=True)
    X.reset_index(drop=True, inplace=True)
    y = X[target_col]
    X.drop(columns=target_col, inplace=True)
    return (X,y)



# %% Vitals
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score,\
    recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold, StratifiedKFold


X,y = get_data(capData, data_cols='lab')
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, y)

split_results = {}
M = X.drop(columns=['enc_id','person_id'])
L = y
for split_id, (train_index, test_index) in enumerate(skf.split(X, y)):
    #clf = LogisticRegression()
    clf = RandomForestClassifier()
    try:  # always set random and  ll processors, w/o confounding param_grid
        clf.set_params(random_state=86)
        clf.set_params(n_jobs=-1)
    except:
        pass
    clf.fit(M.iloc[train_index],L.iloc[train_index])

    preds = clf.predict(M.iloc[test_index])
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

print(final_results)

# Get Feature Importances
iDF = pd.DataFrame(zip(M,clf.feature_importances_), 
columns=['Feature','Feature_Importance']).sort_values(
    by='Feature_Importance', ascending=False)


iDF.nlargest(20, 'Feature_Importance')

# %%
iDF = pd.DataFrame(zip(M,clf.feature_importances_), 
columns=['Feature','Feature_Importance']).sort_values(
    by='Feature_Importance', ascending=False)




# %%

import re
def find_diag(lookup, return_val='description'):
    print('Terms = ', lookup) 
    result = list(capData.diagnosis[capData.diagnosis.description.str.contains(lookup, regex=True, flags=re.IGNORECASE)][return_val].unique())
    return result
# %%

diag_codes = (code.lstrip('asmt_icd_') for code in iDF.nlargest(20, 'Feature_Importance').Feature)
descriptions = find_diag('|'.join(diag_codes))

print(descriptions)
list(zip(diag_codes, descriptions))

# %%
[code for code in iDF.Feature]

# %%
capData
# %%
