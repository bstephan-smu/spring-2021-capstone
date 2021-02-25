
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_data import DataLoader
# %%
capData = DataLoader().load()
# %%
df=pd.DataFrame(capData.main.copy())

# %%
import datetime as dt
# %%
df= df.groupby('person_id').apply(lambda x: x.sort_values('enc_EncounterDate')).reset_index(drop=True)

# %% 
# change ordinal to y,m,d
df['enc_EncounterDate']= df['enc_EncounterDate'].astype(int).map(dt.datetime.fromordinal)
# %%
for col in df.columns: 
    print(col)
# %%
# drop nan for gender and sex 

df.dropna(subset=['enc_Gender_F', 'enc_Gender_M', 'enc_AgeAtEnc'], how='all')
# %% 
pd.set_option('display.max_rows', None)
df.isnull().sum().sort_values(ascending = False)

# %% 

df.head()
# %%
visit_set= df.groupby('person_id').apply(lambda x: x.sort_values('enc_EncounterDate'))
# %%
visit_set
    
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
df.count()
# %%
df.set_index(["person_id", "enc_Race_Native Hawaiian or Other Pacific Islander"]).count(level="person_id")
# %%
for col in df.columns: 
    print(col)

# %%
# subset the data based wellness cpt codes
#well_code=['cpt_G0438', 'cpt_G0439']

wellness=df[(df.cpt_G0438 ==1) |( df.cpt_G0439)]
# %%
wellness.shape
# %%
wellness.head()
# %%
wellness.isnull().sum().sort_values(ascending = False)
# %%
wellness.count()
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
df1 = wellness.sort_values(['person_id', 'enc_EncounterDate'], ascending=[True, True])
df1['visit'] = df1.groupby(['person_id'])['enc_EncounterDate'].rank(ascending=True, method='dense').astype(int)
cols=df1.columns.tolist()
cols = cols[-1:] + cols[:-1]
df1=df1[cols]
# %%
df1

# %%
df1.head()



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
enc['race_ethincity'] = enc['Race'] + enc['Ethnicity']
# %%
enc

# %%
 enc['race_test']=enc['race_ethincity'].replace([
     #concatinated column values 
    'White Declined To Specify',  
    'NULL NULL' , 
    'White Hispanic or Latino',
    'White Not Hispanic or Latino', 
    'Black or African American Not Hispanic or Latino',
    'Declined To Specify Not Hispanic or Latino',
    'White Unknown / Not Reported',
    'Declined to Specify Not Hispanic or Latino', 
    'Asian Declined To Specify',
    'Declined to Specify Declined To Specify',
    'Black or African American Declined To Specify',
    'American Indian or Alaska Native Not Hispanic or Latino',
    'White Declined to Specify',
    'Asian Not Hispanic or Latino',
    'Native Hawaiian or Other Pacific Islander Not Hispanic or Latino', 
    'Declined To Specify Declined To Specify',  
    'Black or African American Unknown / Not Reported',
    'Black or African American Hispanic or Latino',  
    'Declined To Specify Hispanic or Latino',  
    'Declined to Specify Unknown / Not Reported', 
    'Black or African American Declined to Specify',     
    'Declined to Specify Hispanic or Latino', 
    'Declined To Specify Declined to Specify', 
    'American Indian or Alaska Native Declined To Specify', 
    'Declined To Specify Unknown / Not Reported', 
    'Asian Unknown / Not Reported',
    'American Indian or Alaska Native Hispanic or Latino',
    'Asian Hispanic or Latino',
    'NA Declined To Specify'],
    #these are the renames for each of the concatinated columns above
    ['White, Not Hispanic or Latino',
    'NULL','White, Hispanic or Latino',
    'White, Not Hispanic or Latino',
    'Black or African American',
    'Declined To Specify',
    'White, Not Hispanic or Latino',
    'Declined To Specify',
    'Asian',
    'Declined To Specify',
    'Black or African American',
    'American Indian or Alaska Native',
    'White, Not Hispanic or Latino',
    'Asian',	
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
    'Declined To Specify']) 

# %%
enc['race_test']=enc['race_ethincity'].replace(['White Declined To Specify'], 'White, Not Hispanic or Latino')

# %% 
enc.head()
# %%
