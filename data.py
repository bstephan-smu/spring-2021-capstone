
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import DataLoader
# %%

from encoder import Encoder

# %%
capData = Encoder().build() # Run to rebuild the pickle after making changes to load_data.py or encoder.py
#capData = Encoder().load()
# %%
df=pd.DataFrame(capData.main.copy())

# %%
#df.head()
#df.shape
for col in df.columns: 
    print(col)
# %%


# %% 
import datetime as dt
# change ordinal to y,m,d
df['enc_EncounterDate']= df['enc_EncounterDate'].astype(int).map(dt.datetime.fromordinal)
# %%
# %%
#g_df= df.groupby('person_id').apply(lambda x: x.sort_values('enc_EncounterDate')).reset_index(drop=True)
# %%
df.head()
# %%
for col in df.columns: 
    print(col)
# %%
#create a subset for wellness exam only



# %% 

pd.set_option('display.max_rows', None)
df.isnull().sum().sort_values(ascending = False)

# %% 


# %%

    
#%%
frequency = df['person_id'].value_counts()


# %%
print(frequency)
# %%
plt.hist(frequency, bins = 100)
plt.show()
# %%
df['person_id'].nunique()

# %%
for col in df.columns: 
    print(col)

# %%
# add visit # startying at 1 to the entire df
df = df.sort_values(['person_id', 'enc_EncounterDate'], ascending=[True, True])
df['visit'] = df.groupby(['person_id'])['enc_EncounterDate'].rank(ascending=True, method='dense').astype(int)
cols=df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df=df[cols]
#%%
# subset the data based wellness cpt codes
#well_code=['cpt_G0438', 'cpt_G0439']
wellness=df[(df['cpt_ANNUAL WELLNES VST; PERSNL PPS INIT'] ==1) | (df['cpt_ANNUAL WELLNESS VST; PPS SUBSQT VST']==1)]
# %%
# add visit # startying at 1 to wellness
wellness = wellness.sort_values(['person_id', 'enc_EncounterDate'], ascending=[True, True])
wellness['visit'] = wellness.groupby(['person_id'])['enc_EncounterDate'].rank(ascending=True, method='dense').astype(int)
cols=wellness.columns.tolist()
cols = cols[-1:] + cols[:-1]
wellness=wellness[cols]

# %%
wellness.head()
# %%
# Drop columns not used for modeling
c_wellness=wellness.drop(['person_id', 'enc_id','enc_EncounterDate',
'Reason_for_Visit',
'AD_encounter',
'AD_person',
'dem_encounter',
'dem_person',
'asmt_icd9cm_code_id',
'asmt_diagnosis_code_id',
'asmt_description',
'asmt_date_onset_sympt',
'asmt_date_diagnosed',
'asmt_date_resolved',
'asmt_status_id',
'asmt_dx_priority',
'asmt_chronic_ind',
'asmt_recorded_elsewhere_ind',
'ccsr_Neurocognitive disorders',

], axis=1)
# %%
c_wellness = c_wellness.loc[:, ~c_wellness.columns.str.startswith('cpt_')]
# %%
for col in c_wellness.columns: 
    print(col)
# %%
year1 = c_wellness[c_wellness['visit']==1]

# %%
year1.dropna(inplace=True)
# %%
year1.shape
# %%
year1=year1.drop(['visit'], axis=1)
year1 = year1[year1.Cognition !='Dementia']


# %%

# %%

pd.set_option('display.max_rows', None)
year1.isnull().sum().sort_values(ascending = False)


# %%
y1= year1['Cognition']

#%%
y1

# %%
y1.Cognition[y1.Cognition =='Normal'] =0
y1.Cognition[y1.Cognition =='AD'] =1

# %%
#New table
y1['Cognition'] = df['Cognition'].replace(['Normal','AD',],[0,1])
# %%
X1=year1.drop(['Cognition'], axis=1)
#  %%
len(X1.columns)
# %%
for col in X1.columns: 
    print(col)

# %%
# Second Year
year2 = c_wellness[c_wellness['visit']==2]

# %%
year2.shape
# %%
year2=year2.drop(['visit'], axis=1)
year2 = year2[year2.Cognition !='Dementia']


#%%
y2= year2['Cognition']
# %%
X2=year2.drop(['Cognition'], axis=1)
#%$
y2

# %%
from sklearn.model_selection import train_test_split
# %%
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=10, random_state=42)
# %%
from sklearn.preprocessing import StandardScaler
# %%
feature_scaler = StandardScaler()
X1_train = feature_scaler.fit_transform(X1_train)
X1_test = feature_scaler.transform(X1_test)
#%%
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=300, random_state=42)
# %%
classifier.fit(X1_train, y1_train)
y_pred = classifier.predict(X1_test)

# %%

from sklearn import metrics
print('Mean Absolute Error:', metrics.accuracy_score(y1_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y1_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y1_test, y_pred)))

# %%
from sklearn.model_selection import cross_val_score
all_accuracies = cross_val_score(estimator=classifier, X=X1_train, y=y1_train, cv=10)
# %%

print(all_accuracies)

# %%
for col in X1.columns: 
    print(col)



# %%

from sklearn import datasets
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.ensemble import  RandomForestClassifier
import pandas as pd
# %%

X1=year1.drop(['Cognition'], axis=1)

# %%

# %%
X1, y1= year1, year1.Cognition
# %%
clf=RandomForestClassifier(n_estimators =20, random_state = 42)
output = cross_validate(clf, X1, y1, cv=5, scoring = 'accuracy', return_estimator =True)
# %%
clf.fit(X1, y1)
# %%

#clf.feature_importances_
clf.score()

# %%

for idx,estimator in enumerate(output['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    feature_importances = pd.DataFrame(estimator.feature_importances_,
                                       index = X1.feature_names,
                                        columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)

# %%
plt.barh(year1.feature_names, rf.feature_importances_)


# %%
#541 Normal
#88 AD
#43 Demenita
wellness[(wellness.Cognition =='Normal')]
wellness[(wellness.Cognition =='AD')]
wellness[(wellness.Cognition =='Dementia')]
# %%
well_frequency = wellness['person_id'].value_counts()
# %%
plt.hist(well_frequency, bins = 20)
plt.show()

# %%

# %%
df.head()

# %%
df1.head()
# %%
df1[(df1.Cognition =='Normal')]
# %%
df1[(df1.Cognition =='AD')] 
# %%
df1[(df1.Cognition =='Dementia')]


# %%

df['visit_year'] = df['enc_EncounterDate'].dt.year

# %%

df.head()
# %%

df_jeff = df.sort_values(['person_id', 'visit_year'], ascending=[True, True])
df_jeff['visit_number'] = df.groupby(['person_id'])['visit_year'].rank(ascending=True, method='dense').astype(int)
cols=df_jeff.columns.tolist()
cols = cols[-1:] + cols[:-1]
df_jeff=df_jeff[cols]

#%%
df_jeff=df_jeff.groupby(['person_id'])
# %%


df_jeff

# %% jeff's code
#main_set['visit_group_num'] = main_set.groupby(['person_id', 'visit_year']).ngroup()

# %%
# First Year
year1 = df_jeff[df_jeff['visit_number']==1]
 
 
 # %%
#541 Normal
#88 AD
#43 Demenita
year1[(year1.Cognition =='Normal')]

# %%
year1[(year1.Cognition =='AD')]
# %%
year1[(year1.Cognition =='Dementia')]

# %%
# Second Year
year2 = df_jeff[df_jeff['visit_number']==2]
# Third Year
year3 = df_jeff[df_jeff['visit_number']==3]
# Fourth Year Plus
year4 = df_jeff[df_jeff['visit_number']>=4]


# %%
year1_frequency = year1['person_id'].value_counts()
plt.hist(year1_frequency, bins = 20)
plt.show()
# %%
year2_frequency = year2['person_id'].value_counts()
plt.hist(year2_frequency, bins = 20)
plt.show()
# %%
year3_frequency = year3['person_id'].value_counts()
plt.hist(year3_frequency, bins = 20)
plt.show()

#%%
year4_frequency = year4['person_id'].value_counts()
plt.hist(year4_frequency, bins = 20)
plt.show()

# %%


for col in df.columns: 
    print(col)
# %%
import pandas as pd
enc = pd.read_csv (r'E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/1_BaseEncounters_Dempgraphics_Payers.csv')

# %%
enc.head()

# %%
enc.drop(columns=['Encounter_Teritiary_Payer'], inplace=True)
# %%
enc.columns = ['person_id', 'enc_id', 'place_of_service', 'Provider_id', 'EncounterDate', 'Race',
                                   'Ethnicity', 'Gender', 'AgeAtEnc', 'VisitType', 'ServiceDepartment',
                                   'LocationName',
                                   'Reason_for_Visit', 'CPT_Code', 'CPT_Code_Seq', 'Encounter_Primary_Payer',
                                   'Encounter_Secondary_Payer', 'Encounter_Teritiary_Payer']
# %%
enc.head()

# %%
enc['r&e'] = enc['Race'] + enc['Ethnicity']

# %%
enc['r&e'] = enc['Race'] + enc['Ethnicity']
enc['race_ethincity']=enc['r&e'].replace([
     #concatinated column values 
    'White Hispanic or Latino ',
    'White Declined To Specify ', 
    'White Not Hispanic or Latino ',
    'Black or African American Not Hispanic or Latino ',
    'Declined To Specify Not Hispanic or Latino ',
    'White Unknown / Not Reported ',
    'Declined to Specify Not Hispanic or Latino ',
    'Asian Declined To Specify ',
    'Declined to Specify Declined To Specify ',
    'Black or African American Declined To Specify ',
    'American Indian or Alaska Native Not Hispanic or Latino ',
    'Asian Not Hispanic or Latino ',
    'White Declined to Specify ', 
    'Native Hawaiian or Other Pacific Islander Not Hispanic or Latino ',
    'Declined To Specify Declined To Specify ',
    'Black or African American Unknown / Not Reported ',
    'Black or African American Hispanic or Latino ',
    'Declined To Specify Hispanic or Latino ',
    'Declined to Specify Unknown / Not Reported ',
    'Black or African American Declined to Specify ',
    'Declined to Specify Hispanic or Latino ',
    'Declined To Specify Declined to Specify ',
    'American Indian or Alaska Native Declined To Specify ',
    'Declined To Specify Unknown / Not Reported ',
    'Asian Unknown / Not Reported ',
    'American Indian or Alaska Native Hispanic or Latino ',
    'Asian Hispanic or Latino ',
    ' Declined To Specify '
],
    #these are the renames for each of the concatinated columns above
    ['White, Hispanic or Latino',				
    'White, Not Hispanic or Latino',				
    'White, Not Hispanic or Latino',				
    'Black or African American',				
    'Declined To Specify',				
    'White Not Hispanic or Latino',				
    'Declined To Specify',				
    'Asian',			
    'Declined To Specify',				
    'Black or African American',				
    'American Indian or Alaska Native',				
    'Asian',				
    'White, Not Hispanic or Latino',				
    'Native Hawaiian or Other Pacific Islander',				
    'Declined To Specify',				
    'Black or African American',				
    'Black or African American, Hispanic or Latino',				
    'Hispanic or Latino',				
    'Declined To Specify',				
    'Black or African American',				
    'Hispanic or Latino',				
    'Declined To Specify',				
    'American Indian or Alaska Native',				
    'Declined To Specify',				
    'Asian',			
    'American Indian or Alaska Native, Hispanic or Latino',				
    'Asian, Hispanic or Latino',				
    'Declined To Specify'				
]) 
enc.drop(['r&e', 'Race', 'Ethnicity'], axis=1)

# %%
enc['race_ethincity'].unique()
# %%
# %%
enc.head()
# %%

g_df= df.groupby('person_id').apply(lambda x: x.sort_values('enc_EncounterDate')).reset_index(drop=True)
# %%

g_df[(g_df.Cognition =='Normal')]
#%%
df[(df.Cognition =='AD')]
df[(df.Cognition =='Dementia')]
# %%
