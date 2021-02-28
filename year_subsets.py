#%% Load All data 
from encoder import Encoder
#capData = Encoder().build() # Run to rebuild the pickle after making changes to load_data.py or encoder.py
capData = Encoder().load()

# %% Assign capData to dataframe object

import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.DataFrame(capData.main.copy())

#%% Create subsets based on date
import datetime as dt

# Convert date to YYYY-MM-DD to allow for subsetting of data by visit
main_set = df.groupby('person_id').apply(lambda x: x.sort_values('enc_EncounterDate')).reset_index(drop=True)
main_set['enc_EncounterDate']= main_set['enc_EncounterDate'].astype(int).map(dt.datetime.fromordinal)
main_set['visit_year'] = main_set['enc_EncounterDate'].dt.year

#%% Subset visits by date

'''Create labels for visits group by the year the patient visisted.  Ie 1st year visit, 2nd, 3rd, 4th+.  Groups within each patient encounters
and is agnostic of actual year a patient visits.
'''
df_jeff = main_set.sort_values(['person_id', 'visit_year'], ascending=[True, True])
df_jeff['visit_number'] = main_set.groupby(['person_id'])['visit_year'].rank(ascending=True, method='dense').astype(int)
cols=df_jeff.columns.tolist()
cols = cols[-1:] + cols[:-1]
df_jeff=df_jeff[cols]


#%% Columns to drop

# Drop columns that are a result of preprocessing, are mostly NA, directly correlated with response, or in date format.
drop_col = ['enc_EncounterDate','Reason_for_Visit','asmt_icd9cm_code_id','asmt_icd9cm_code_id','asmt_description','asmt_date_onset_sympt',
            'asmt_date_diagnosed','asmt_date_resolved','asmt_status_id','asmt_dx_priority','asmt_chronic_ind','asmt_recorded_elsewhere_ind',
            'ccsr_Neurocognitive disorders']

#Drop unneeded columns
df2 = df_jeff.drop(drop_col, axis=1)

#%% Identify AD/Dementia positive patients to find features with highest frequency of occurance

ad = df2[df2['Cognition'].str.contains("AD|Dementia")]
ad_cpt = ad.filter(regex='cpt')
ad_med = ad.filter(regex='med_')
ad_lab = ad.filter(regex='lab')
ad_ccsr = ad.filter(regex='ccsr')

#%% take a look inside AD patients

'''
Identify features within their tables that have the highest frequency for patients that are AD positive.  Drive model focus towards
features that are most common for AD/Dementia Patients.  Features that have counts greater than 10% of total records are retained
'''
#CPT Codes
ad_cpt_col = pd.DataFrame(ad_cpt.sum().sort_values(ascending=False).head(25))
ad_cpt_col.reset_index(level=0, inplace=True)
'''
cpt_OFFICE/OUTPATIENT VISIT EST	6507
cpt_ANNUAL DEPRESSION SCREENING 15 MIN	608
cpt_OFFICE/OUTPATIENT VISIT NEW	438
cpt_ADMINISTRATION INFLUENZA VIRUS VACC	404
cpt_AMNT PAIN NOTED NONE PRSNT	358
cpt_DIAST BP <80 MM HG	344
cpt_SYST BP LT 130 MM HG	236
cpt_IIV NO PRSV INCREASED AG IM	218
cpt_AMNT PAIN NOTED PAIN PRSNT	211
cpt_ADMINISTRATION PNEUMOCOCCAL VACC	182
cpt_SYST BP >/= 140 MM HG	177
cpt_IIV ADJUVANT VACCINE IM	177
cpt_DIAST BP 80-89 MM HG	174
cpt_SYST BP GE 130 - 139MM HG	159
cpt_PCV13 VACCINE IM	152
cpt_ANNUAL WELLNESS VST; PPS SUBSQT VST	127
cpt_ELECTROCARDIOGRAM COMPLETE	126
cpt_HG A1C LEVEL LT 7.0%	125
cpt_MED LIST DOCD IN RCRD	124
cpt_THERAPEUTIC EXERCISES	110
cpt_GLYCOSYLATED HEMOGLOBIN TEST	97
cpt_THER/PROPH/DIAG INJ SC/IM	77
cpt_MOB: WALK MOV AROUND FCN LIM GOAL	65
cpt_REMOVE IMPACTED EAR WAX UNI	58
cpt_DIAST BP >/= 90 MM HG	53
'''
#Med Codes
ad_med_col = pd.DataFrame(ad_med.sum().sort_values(ascending=False).head(31))
ad_med_col.reset_index(level=0, inplace=True)
'''
ccsr_Otitis media	16
ccsr_Adverse effects of drugs and medicaments, initial encounter	8
ccsr_Diseases of middle ear and mastoid (except otitis media)	8
ccsr_Complication of other surgical or medical care, injury, initial encounter	4
med_aspirin	4
med_aspirin low dose	3
med_multivitamin tablet	2
med_lisinopril	2
med_carvedilol	2
med_vesicare	2
med_metoprolol tartrate	1
med_memantine	1
med_loratadine	1
med_escitalopram	1
med_lipitor	1
med_glipizide er	1
med_glipizide	1
med_fluticasone	1
med_fexofenadine	1
med_clonazepam	1
med_dorzolamide	1
med_divalproex	1
med_clopidogrel	1
med_omega	1
med_clindamycin	1
'''

#Lab codes
labs = pd.DataFrame(df2.filter(regex='lab_').columns)
'''Labs data are mostly NA'''

# ccsr codes
ad_ccsr_col = pd.DataFrame(ad_ccsr.sum().sort_values(ascending=False).head(80))
ad_ccsr_col.reset_index(level=0, inplace=True)

# rfc_cluster
rfc_cluster = pd.DataFrame(df2.filter(regex='rfv').columns)


#asmt_cluster
asmt_cluster = pd.DataFrame(df2.filter(regex='asmt_kmeans|asmt_topic').columns)


#%% Create year subsets

import numpy as np

#Identify columns that are part of feature space
cpt_ccsr_med_col = list(ad_ccsr_col['index']) + list(ad_cpt_col['index']) + list(ad_med_col['index'])
cols = ['Cognition','visit_number','enc_AgeAtEnc']
clusters = list(rfc_cluster[0]) + list(asmt_cluster[0])

# Recode response to binary 0 = Normal, 1 = AD/Dementia
df2['Cognition']= df2['Cognition'].replace(to_replace="AD",value= 1)
df2['Cognition']= df2['Cognition'].replace(to_replace="Dementia",value= 1)
df2['Cognition']= df2['Cognition'].replace(to_replace="Normal",value= 0)

# Create dataset that contains features and response to allow for subsetting by year
features = cols + cpt_ccsr_med_col + clusters + list(labs[0])
df2 = df2[features]

year1 = df2[df2['visit_number']==1]
year2 = df2[df2['visit_number']==2]
year3 = df2[df2['visit_number']==3]
year4 = df2[df2['visit_number']==4]


#%% Create train/test sets

# Create inital feature space and response for all data, fill any NAs in feature space with 0's
# Remove Cognition from feature space

y = df2['Cognition'].values
features.remove('Cognition')
feature_space = df2[features]
X = feature_space.fillna(0).values

'''Create holdout set for modeling, include sets for all data, year1, year2, year3, year 4+ '''
from sklearn.model_selection import train_test_split

#all Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# year 1 data
X1 = year1[features].fillna(0).values
y1 = year1['Cognition'].values
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.20, random_state=42)


# year 2 data
X2 = year2[features].fillna(0)
y2 = year2['Cognition'].values
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.20, random_state=42)


# year 3 data
X3 = year3[features].fillna(0).values
y3 = year3['Cognition'].values
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.20, random_state=42)


# Year 4+ data
X4 = year4[features].fillna(0)
y4 = year4['Cognition'].values
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.20, random_state=42)

#%%Create Subsets and oversmaple minorty class

'''
Upsample each subset to improve model training and overall model preformance using SMOTE
'''
from imblearn.over_sampling import SMOTE
sm = SMOTE()

# All data
X_sm, y_sm = sm.fit_resample(X_train, y_train)

# First Year
X1_sm, y1_sm = sm.fit_resample(X1_train, y1_train)

# # Second Year
X2_sm, y2_sm = sm.fit_resample(X2_train, y2_train)

# Third Year
X3_sm, y3_sm = sm.fit_resample(X3_train, y3_train)

# Fourth Year Plus
X4_sm, y4_sm = sm.fit_resample(X4_train, y4_train)



#%% Create Metrics
'''
Create metrics that are commonly referenced in AD/Dementia Research
'''

from sklearn import metrics as mt
def npv_score(ytest, yhat):
    # Negative Predictive Value identifies the probability that a patient is actually negative for a test
    confusion = mt.confusion_matrix(yhat, ytest)

    TN = confusion[0, 0]
    FN = confusion[1, 0]

    # convert to float and Calculate score 
    npv = (TN.astype(float)/(FN.astype(float)+TN.astype(float)))
    #    (True Negative/(Predicted Negative + True Negative))
    
    return npv

def ppv_score(ytest, yhat):
     # Posotive Predictive Value identifies the probability that a patient is actually positive for a test
    confusion = mt.confusion_matrix(yhat, ytest)
    FP = confusion[0, 1]
    TP = confusion[1, 1]

    # convert to float and Calculate score 
    ppv = (TP.astype(float)/(FP.astype(float)+TP.astype(float)))
    #    (True Negative/(Predicted Negative + True Positive))
    
    return ppv


def specificity_score(ytest, yhat):
    confusion = mt.confusion_matrix(yhat, ytest)
    TN = confusion[0, 0]
    FP = confusion[0, 1]

    spec = TN.astype(float)/(FP.astype(float) + TN.astype(float))

    return spec


#%% Create Score Model Helper Function

'''
Helper function provides model preformance metrics for one interation as a means to identify baseline preformance
'''
import pandas as pd
from sklearn import metrics as mt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def score_model(model, X_sm, X_test, y_sm, y_test, feature_names = df2[features].columns):
      
    clf = model.fit(X_sm,y_sm)  # train object
    y_hat= clf.predict(X_test) # get test set precitions
    
    #present iteration model metrics
    print("roc_auc", mt.roc_auc_score(y_test,y_hat))
    print("Positive Predictive Value", ppv_score(y_test,y_hat))
    print("Negative Predictive Value", npv_score(y_test,y_hat))
    print("f1-score", mt.f1_score(y_test,y_hat))
    print("Accuracy", mt.accuracy_score(y_test,y_hat))
    print("Precision", mt.precision_score(y_test,y_hat))
    print("Sensitivity/Recall", mt.recall_score(y_test,y_hat))
    print("Specificity", specificity_score(y_test,y_hat))
    print("\nconfusion matrix\n",mt.confusion_matrix(y_test,y_hat))
    print()
    

#%% Logistic Regression  initial models to see what preformance might look like, uncomment line to see what scores are like for a parituclar year
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(penalty = 'l2',n_jobs = -1, solver = 'lbfgs')

#Year 1
#score_model(clf, X1_sm, X1_test, y1_sm, y1_test)

#Year 2
#score_model(clf, X2_sm, X2_test, y2_sm, y2_test)

#Year 3
#score_model(clf, X3_sm, X3_test, y3_sm, y3_test)

#Year 4
#score_model(clf, X4_sm, X4_test, y4_sm, y4_test)


# %% Random Forest initial models to see what preformance might look like, uncomment line to see what scores are like for a parituclar year

rfc = RandomForestClassifier(max_depth=200, n_estimators=500, n_jobs=-1, oob_score=True,criterion = 'entropy')

#Year 1
#score_model(rfc, X1_sm, X1_test, y1_sm, y1_test)

#Year 2
#score_model(rfc, X2_sm, X2_test, y2_sm, y2_test)

#Year 3
#score_model(rfc, X3_sm, X3_test, y3_sm, y3_test)

#Year 4
# #score_model(rfc, X4_sm, X4_test, y4_sm, y4_test)


#%% xgboost not currently fitted

from sklearn.ensemble import GradientBoostingClassifier
xgb = GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, max_depth=1, random_state=0)


#%% Gridsearch helper function
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
import time


cv = StratifiedKFold(n_splits=2,shuffle=True, random_state=42)
def gridsearch(model, grid, rf, scoring, x_test, y_test, X_train, y_train,modelType = 0, cv=cv):
    # model - input model object
    # grid - parameters used for gridsearch
    # rf - refit parameter for gridsearch aka performance metric
    # scoring - metric of choice for model preformance
    # x_test - testing df
    # X_train - features
    # y_train - target
    # cv - cv object
    
    start = time.time() #Start timer
    #Preform Grid Search and get predictions
    cv_results = GridSearchCV(model, grid, cv=cv, scoring = scoring, refit = rf, return_train_score = True, n_jobs=-1)
    cv_results.fit(X_train,y_train)  # train object
    y_hat = cv_results.predict(x_test)
    elapsed_time = (time.time() - start)  #end timer
    

    #print out preformance metics and confusion matrix for classification
    #########################################################
    print("roc_auc", mt.roc_auc_score(y_test,y_hat))
    print("Positive Predictive Value", ppv_score(y_test,y_hat))
    print("Negative Predictive Value", npv_score(y_test,y_hat))
    print("f1-score", mt.f1_score(y_test,y_hat))
    print("Accuracy", mt.accuracy_score(y_test,y_hat))
    print("Precision", mt.precision_score(y_test,y_hat))
    print("Sensitivity/Recall", mt.recall_score(y_test,y_hat))
    print("Specificity", specificity_score(y_test,y_hat))
    print()
    print('Grid Search Time: ', elapsed_time)
    print("==== Confusion Matrix ====")
    print("\nconfusion matrix\n",mt.confusion_matrix(y_test,y_hat))
    print()
 
    return [cv_results.best_estimator_,cv_results.best_score_, cv_results.best_params_, cv_results.cv_results_] #return grid search results

#%% Random Forest Gridsearch 
################################################################################################################################

#initialize model and cv objects
rfc = RandomForestClassifier(n_jobs=-1, oob_score=True)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#set up parameters and metrics to tune hyperparameters by
grid = {'max_features':['auto','sqrt'],
        'min_samples_split': [2,4,6,8,10,20,30,40,50,60,70,100,200,300,400,500],
        'n_estimators': [100, 200, 300]}
rf = "npv"
scoring = {'recall':make_scorer(mt.recall_score),
            'npv': make_scorer(npv_score),
            'ppv':make_scorer(ppv_score)
            }

#execute grid search using RF on 1st years
gridsearch(rfc,grid, rf, scoring, X1_test, y1_test, X1_sm, y1_sm, cv=cv)
#%% RF GS on 2nd years
gridsearch(rfc,grid, rf, scoring, X2_test, y2_test, X2_sm, y2_sm, cv=cv)
#%% RF GS on 3rd years
gridsearch(rfc,grid, rf, scoring, X3_test, y3_test, X3_sm, y3_sm, cv=cv)
#%% RF GS on 4th years
gridsearch(rfc,grid, rf, scoring, X4_test, y4_test, X4_sm, y4_sm, cv=cv)


# %% Logistic Grid Search 
################################################################################################################################

#initialize model and cv objects
clf = LogisticRegression(penalty = 'l1',n_jobs = -1)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#set up parameters and metrics to tune hyperparameters by
grid = {'solver':['liblinear','saga'],
        'C': [0.01, 0.1, 0.25, 0.5, 0.75, 1]}
rf = "npv"
scoring = {'npv': make_scorer(npv_score),
            'ppv':make_scorer(ppv_score)}

#execute grid search on first years1
gridsearch(clf,grid, rf, scoring, X1_test, y1_test, X1_sm, y1_sm, cv=cv)

#%% execute grid search on second years
gridsearch(clf,grid, rf, scoring, X2_test, y2_test, X2_sm, y2_sm, cv=cv)

#%% execute grid search on second years
gridsearch(clf,grid, rf, scoring, X3_test, y3_test, X3_sm, y3_sm, cv=cv)

#%% execute grid search on second years
gridsearch(clf,grid, rf, scoring, X4_test, y4_test, X4_sm, y4_sm, cv=cv)

