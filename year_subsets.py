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

#%%Create Subsets and oversmaple minorty class

from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy='not majority')


# First Year
year1 = df_jeff[df_jeff['visit_number']==1]
X1_sm, y1_sm = sm.fit_sample(X, y)# correct X and y for response and feature space

# Second Year
year2 = df_jeff[df_jeff['visit_number']==2]
X2_sm, y2_sm = sm.fit_sample(X, y)

# Third Year
year3 = df_jeff[df_jeff['visit_number']==3]
X3_sm, y3_sm = sm.fit_sample(X, y)

# Fourth Year Plus
year4 = df_jeff[df_jeff['visit_number']>=4]
X4_sm, y4_sm = sm.fit_sample(X, y)




#lab_Hemoglobin (g/dL)
#enc_AgeAtEnc
#lab_flags_High Protein,Total,Urine
#med_memantine