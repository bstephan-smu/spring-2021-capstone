#%% Load All data 
from load_data import DataLoader
capData = DataLoader().load()


# %% take a look inside the df
import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.DataFrame(capData.main.copy())

#%% Create subsets
import datetime as dt

main_set = df.groupby('person_id').apply(lambda x: x.sort_values('enc_EncounterDate')).reset_index(drop=True)
main_set['enc_EncounterDate']= main_set['enc_EncounterDate'].astype(int).map(dt.datetime.fromordinal)
main_set['visit_year'] = main_set['enc_EncounterDate'].dt.year

#%% Subset visits

df_jeff = main_set.sort_values(['person_id', 'visit_year'], ascending=[True, True])
df_jeff['visit_number'] = main_set.groupby(['person_id'])['visit_year'].rank(ascending=True, method='dense').astype(int)
cols=df_jeff.columns.tolist()
cols = cols[-1:] + cols[:-1]
df_jeff=df_jeff[cols]




#%% Columns to drop

drop_col = ['enc_EncounterDate','Reason_for_Visit','asmt_txt_description','asmt_txt_tokenized','asmt_ngrams',
            'asmt_ngram2','asmt_txt_tokenized2','asmt_np_chunks','asmt_description','CCSR Category','asmt_txt_diagnosis_code_id',
            'asmt_np_chunk_clusters','asmt_topic_clusters','asmt_icd9cm_code_id','asmt_diagnosis_code_id','asmt_date_onset_sympt',
            'asmt_date_diagnosed','asmt_date_resolved','asmt_status_id','asmt_dx_priority','asmt_chronic_ind',
            'asmt_recorded_elsewhere_ind','asmt_CCSR Category','ccsr_NVS011','asmt_kmeans_9']

#Drop unneeded columns
df2 = df_jeff.drop(drop_col, axis=1)

#%% EDA

ad = df2[df2['Cognition'].str.contains("AD|Dementia")]
ad_cpt = ad.filter(regex='cpt')
ad_med = ad.filter(regex='med')
ad_lab = ad.filter(regex='lab')
ad_ccsr = ad.filter(regex='ccsr')

#%% take a look inside AD patients

#CPT Codes
ad_cpt_col = pd.DataFrame(ad_cpt.sum().sort_values(ascending=False).head(25))
ad_cpt_col.reset_index(level=0, inplace=True)
'''
cpt_99214    4296
cpt_99213    1604
cpt_G0444     608
cpt_99215     546
cpt_G0008     404
cpt_1126F     358
cpt_3078F     344
cpt_3074F     236
cpt_99203     223
cpt_90662     218
cpt_1125F     211
cpt_G0009     182
cpt_3077F     177
cpt_90653     177
cpt_3079F     174
cpt_3075F     159
cpt_90670     152
cpt_99204     131
cpt_G0439     127
cpt_93000     126
cpt_3044F     125
cpt_1159F     124
cpt_97110     110
cpt_83036      97
cpt_96372      77
'''
#Med Codes
ad_med_col = pd.DataFrame(ad_med.sum().sort_values(ascending=False).head(25))
ad_med_col.reset_index(level=0, inplace=True)
'''
med_aspirin                      4.0
med_aspirin low dose             3.0
med_carvedilol                   2.0
med_lisinopril                   2.0
med_vesicare                     2.0
med_multivitamin tablet          2.0
med_norvasc                      1.0
med_divalproex                   1.0
med_isosorbide mononitrate er    1.0
med_bystolic                     1.0
med_omega                        1.0
med_fluticasone                  1.0
med_plavix                       1.0
med_metoprolol tartrate          1.0
med_memantine                    1.0
med_clindamycin                  1.0
med_loratadine                   1.0
med_lipitor                      1.0
med_clopidogrel                  1.0
med_dorzolamide                  1.0
med_potassium chloride er        1.0
med_amiodarone                   1.0
med_fexofenadine                 1.0
med_glipizide                    1.0
med_glipizide er                 1.0
'''

#Lab codes
ad_lab.count().sort_values(ascending=False).head(25)

'''
Labs data is mostly NA
'''

# ccsr codes
ad_ccsr_col = pd.DataFrame(ad_ccsr.sum().sort_values(ascending=False).head(50))
ad_ccsr_col.reset_index(level=0, inplace=True)
'''
ccsr_CIR007    2965.0
ccsr_NVS011    2566.0
ccsr_END005    1383.0
ccsr_END010    1263.0
ccsr_FAC025    1028.0
ccsr_SYM010     988.0
ccsr_END002     822.0
ccsr_MUS010     816.0
ccsr_SYM016     791.0
ccsr_MBD002     768.0
ccsr_END003     640.0
ccsr_END001     638.0
ccsr_GEN003     576.0
ccsr_MUS006     543.0
ccsr_FAC014     512.0
ccsr_MBD005     487.0
ccsr_END009     484.0
ccsr_NVS016     458.0
ccsr_FAC021     450.0
ccsr_DIG004     403.0
ccsr_MUS038     392.0
ccsr_FAC012     387.0
ccsr_RSP008     386.0
ccsr_INJ031     384.0
ccsr_RSP007     379.0
ccsr_FAC016     375.0
ccsr_SYM013     368.0
ccsr_CIR017     366.0
ccsr_NVS019     364.0
ccsr_SYM017     329.0
ccsr_SYM006     324.0
ccsr_END007     320.0
ccsr_CIR011     314.0
ccsr_SKN007     304.0
ccsr_DIG025     284.0
ccsr_MUS011     282.0
ccsr_GEN008     272.0
ccsr_CIR019     271.0
ccsr_MUS013     269.0
ccsr_SYM011     259.0
ccsr_EAR006     254.0
ccsr_SYM007     240.0
ccsr_SYM012     239.0
ccsr_NVS006     228.0
ccsr_NVS015     226.0
ccsr_GEN012     207.0
ccsr_EAR004     202.0
ccsr_GEN004     200.0
ccsr_FAC010     181.0
ccsr_BLD003     170.0
'''

#%% Create df with columns of interest
import numpy as np


cpt_ccsr_med_col = list(ad_ccsr_col['index']) + list(ad_cpt_col['index']) + list(ad_med_col['index'])
cols = ['visit_number','enc_AgeAtEnc']
more_cols = list(df2.iloc[:, 1605:].columns)

df2['Cognition']= df2['Cognition'].replace(to_replace="AD",value= 1)
df2['Cognition']= df2['Cognition'].replace(to_replace="Dementia",value= 1)
df2['Cognition']= df2['Cognition'].replace(to_replace="Normal",value= 0)


features = cols +cpt_ccsr_med_col  + more_cols
y = df2['Cognition'].values
feature_space = df2[features]
X = feature_space.fillna(0).values
#%%Create Subsets and oversmaple minorty class
from imblearn.over_sampling import SMOTE
sm = SMOTE()

# First Year
year1 = df2[df2['visit_number']==1]
y1 = year1['Cognition']
X1 = year1[features].fillna(0)
X1_sm, y1_sm = sm.fit_resample(X1, y1)# correct X and y for response and feature space

# # Second Year
# year2 = df2[df2['visit_number']==2]
# X2_sm, y2_sm = sm.fit_sample(X, y)

# # Third Year
# year3 = df2[df2['visit_number']==3]
# X3_sm, y3_sm = sm.fit_sample(X, y)

# # Fourth Year Plus
# year4 = df2[df2['visit_number']>=4]
# X4_sm, y4_sm = sm.fit_sample(X, y)


 

#lab_Hemoglobin (g/dL)
#enc_AgeAtEnc
#lab_flags_High Protein,Total,Urine
#med_memantine

#%% Make a scoreer
from sklearn import metrics as mt
def npv_score(ytest, yhat):
    confusion = mt.confusion_matrix(yhat, ytest)

    TN = confusion[0, 0]
    FN = confusion[1, 0]

    # convert to float and Calculate score 
    npv = (TN.astype(float)/(FN.astype(float)+TN.astype(float)))
    #    (True Negative/(Predicted Negative + True Negative))
    
    return npv

def ppv_score(ytest, yhat):

    confusion = mt.confusion_matrix(yhat, ytest)
    FP = confusion[0, 1]
    TP = confusion[1, 1]

    # convert to float and Calculate score 
    ppv = (TP.astype(float)/(FP.astype(float)+TP.astype(float)))
    #    (True Negative/(Predicted Negative + True Positive))
    
    return ppv

#%% Helper function
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as mt
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import time

# Set test/train split to 80/20
cv = ShuffleSplit(n_splits = 3, train_size  = 0.8, random_state=42)

#helper function preforms cross validation and generates model preformance metrics
def cross_validate(model,feature_names, X, y, scale = False, classifier = 0, cv=cv):
    auc,acc, ppv, npv, f1, recall, weights = ([] for i in range(7))
        
    start = time.time()
    for iter_num, (train_indices, test_indices) in enumerate(cv.split(X,y)):
        
        model.fit(X[train_indices],y[train_indices])  # train object
        y_hat= model.predict_proba(X[test_indices]) # get test set precitions
        
        prob = 0.40
        y_hat[y_hat >= prob] = 1
        y_hat[y_hat < prob] = 0
        y_hat =  y_hat[:, 1]
        print(y_hat)
        #present iteration model metrics
        print("====Iteration",iter_num + 1," ====")
        print("roc_auc", mt.roc_auc_score(y[test_indices],y_hat))
        print("Positive Predictive Value", ppv_score(y[test_indices],y_hat))
        print("Negative Predictive Value", npv_score(y[test_indices],y_hat))
        print("f1-score", mt.f1_score(y[test_indices],y_hat))
        print("Accuracy", mt.accuracy_score(y[test_indices],y_hat))
        print("Recall", mt.recall_score(y[test_indices],y_hat))
        print("\nconfusion matrix\n",mt.confusion_matrix(y[test_indices],y_hat))
        
        #append iteration model metrics for later averaging
        auc.append(mt.roc_auc_score(y[test_indices],y_hat))
        f1.append(mt.f1_score(y[test_indices],y_hat))
        acc.append(mt.accuracy_score(y[test_indices],y_hat))
        ppv.append(ppv_score(y[test_indices],y_hat))
        npv.append(npv_score(y[test_indices],y_hat))
        recall.append(mt.recall_score(y[test_indices],y_hat))
        elapsed_time = (time.time() - start)
        
        if scale == True:
            weights.append(model.named_steps[classifier].coef_)#logit_model
        else:
            weights.append(model.coef_)
        print()
    #Take average of CV metrics
    print('--------------------------------')
    print('Mean AUC score:', np.array(auc).mean())
    print('Mean Positive Predictive Value:', np.array(ppv).mean())
    print('Mean Negative Predictive Value:', np.array(npv).mean())
    print('Mean f1-score:', np.array(f1).mean())
    print('Mean Accuracy:', np.array(acc).mean())
    print('Mean Recall:', np.array(recall).mean())
    print('CV Time: ', elapsed_time)
        
    mean_feature_weights = np.mean(np.array(weights), axis = 0)
    
    w = pd.Series(mean_feature_weights[0],index=feature_names)
    plt.figure(figsize=(10,6))
    w.plot(kind='bar')
    plt.show()

    pd.set_option('display.max_rows', None)
    d = {'features' : feature_names, 'weights':mean_feature_weights[0]}
    featureWeights = pd.DataFrame(d)
    print(featureWeights)

    return
#%% Inital Modeling

clf = LogisticRegression(penalty = 'l1',n_jobs = -1, solver = 'liblinear')
#cross_validate(clf,feature_space.columns, X, y,scale =False)
cross_validate(clf,feature_space.columns, X1_sm.values, y1_sm.values,scale =False)


# %% Random Forest

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=50, n_estimators=150, n_jobs=-1, oob_score=True)
cross_validate(rfc,feature_space.columns, X1_sm.values, y1_sm.values,scale =False)
