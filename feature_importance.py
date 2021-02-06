#%%
from load_data import DataLoader
capData = DataLoader()
capData.create(name='main')
#capData = capData.load('clean')
#capData.encode_encounters()
#capData.clean()

# %%
def get_data(self, data_cols='all', target_col='AD_event', alt_data=None):
    """
    \ndata_cols: pass the table prefix to return only a table from main options: 
    \n{'cpt_','lab_','vit_','medid_','asmt_icd', 'enc_'}
    \ntarget_col: pass AD_event or dem_event for target
    \ne.g. X, y = get_data(capData, data_cols='vit_')
    """
    df = self.main.copy()
    column_list = ['enc_id','person_id', target_col]# + alt_data

    if alt_data != None:
        column_list += alt_data

    dropcols = [col for col in df if col.lstrip('asmt_icd_') in self.dementia_icd_codes]
    X = df.drop(columns=dropcols)
    
    if data_cols != 'all':
        if data_cols.startswith('enc'):
            column_list = ['person_id', target_col]
        column_list += [col for col in X if col.startswith(data_cols)]
        
    X = X[column_list]

    #X.dropna(inplace=True)
    X.fillna(0, inplace=True)
    X.reset_index(drop=True, inplace=True)
    y = X[target_col]
    X.drop(columns=target_col, inplace=True)
    return (X,y)



# %% Get Feature Importance

def get_feature_importance(dataloader=capData, table=None, target='dem_event', alt_data=None):

    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score,\
        recall_score, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier
    #from sklearn.linear_model import LogisticRegression
    #from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold


    X,y = get_data(dataloader, data_cols=table, target_col=target, alt_data=alt_data)
    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(X, y)

    split_results = {}
    M = X.drop(columns=['enc_id','person_id'])
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

    print(table, final_results)

    # Get Feature Importances
    iDF = pd.DataFrame(zip(M,clf.feature_importances_), 
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

# %%



# %% get all feature importance
FI_enc = get_feature_importance(capData, table='enc_')
FI_vit = get_feature_importance(capData, table='vit_')
FI_med = get_feature_importance(capData, table='medid_')
FI_diag = get_feature_importance(capData, table='asmt_icd_')
FI_cpt = get_feature_importance(capData, table='cpt_')
FI_lab = get_feature_importance(capData, table='lab_')


import pandas as pd
cols = [FI_enc, FI_vit, FI_med, FI_diag, FI_cpt, FI_lab]
all_df = pd.DataFrame()
for df in cols:
    tab = df.Feature[0][:3]
    df.to_csv('FI_'+tab+'.csv')
    all_df = pd.concat([all_df, df[df['Feature_Importance'] > .001]])

FI_all = get_feature_importance(capData, alt_data=list(all_df.Feature), table='all', target='dem_person')
FI_all.to_csv('FI_all.csv')

FI_all.nlargest(30, 'Feature_Importance')

# %%
