#%% Requirements:
import pandas as pd
import numpy as np

# %% Load data:
data_path = 'E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/'

# Fix header on encounters:
new_header = ['person_id','enc_id','place_of_service','Provider_id','EncounterDate','Race','Ethnicity','Gender','AgeAtEnc','VisitType','ServiceDepartment','LocationName','Reason_for_Visit','CPT_Code','CPT_Code_Seq','Encounter_Primary_Payer','Encounter_Secondary_Payer','Encounter_Teritiary_Payer']
encounters = pd.read_csv(data_path + '1_BaseEncounters_Dempgraphics_Payers.csv', header=None, names=new_header, skiprows=1)

# Drop columns with only one value:
encounters = encounters.drop(columns=['place_of_service', 'CPT_Code_Seq'])

cpt_codes = pd.read_csv(data_path + '2_CPT_Codes.csv')
vitals = pd.read_csv(data_path + '3_vitals_signs.csv')
meds = pd.read_csv(data_path + '4_patient_medication.csv')
labs = pd.read_csv(data_path + '5_lab_nor__lab_results_obr_p__lab_results_obx.csv')
diagnoses = pd.read_csv(data_path + '6_patient_diagnoses.csv')

cpt_codes.head()
vitals.head()
meds.head()
labs.head()

# Diagnosis Exploratory Data Analysis
diagnoses.head(10)
# Identify Unique ICD10 codes
#icd9_code = diagnoses.icd9cm_code_id.value_counts()
len(diagnoses['diagnosis_code_id'].unique())

## Word cloud on all descriptions
# Requirements
import wordcloud
import matplotlib.pyplot as plt
import nltk

#nltk.download()

text3 = ' '.join(diagnoses['description'])
wordcloud2 = wordcloud.WordCloud().generate(text3)
# Generate plot
plt.figure(figsize=(20,10))
plt.imshow(wordcloud2)
plt.axis("off")
plt.show()

## Identify all G series ICD10 codes
#G series codes are neurogenitive diseases.  The hope is to identify high frequency words that 
# appear in the disease descriptions and see if they appear with other ICD10 codes to identify 
# possible cases that are diagnosed with other codes. F01, F02, F03 series covers dementia

# Filter ICD10 codes for the G's (neurgentive diseases) and f01-03 (Dementia)

ngd = diagnoses[diagnoses['icd9cm_code_id'].str.contains('G30|F01|F02|F03',case = False)]
diagnoses['icd9cm_code_id'].str.contains('G30|F01|F02|F03',case = False).value_counts()

ngd.reset_index(drop=True)
ngd.head()

#**There are at least 4531 records that have patients with Alzhimers Disease or Dementia ICD10 codes.**
ngd.info()

#G30 codes are for Alzheimer's Disease.

text4 = ' '.join(ngd['description'])
wordcloud3 = wordcloud.WordCloud().generate(text4)
wordcloud3
# Generate plot
plt.figure(figsize=(20,10))
plt.imshow(wordcloud3)
plt.axis("off")
plt.show()

## Joining Data
# see if I can identify a record from ngh in cpt table
ngd[ngd['enc_id'] == '9A3E98A3-14E0-408C-9D98-87D6D7858696']

# see if I can identify a record from ngh in cpt table
cpt_codes[cpt_codes['enc_id'] == '9A3E98A3-14E0-408C-9D98-87D6D7858696']

# See how many records there are
cpt_codes.info()


### Flattening CPT Table
# this method helps identify the maximum number of visits a patient has made.  It also helps identify unique enc_id codes
visit_count = cpt_codes.groupby('enc_id').max()
visit_count.reset_index(level=0, inplace=True)
visit_count.rename(columns={'CPT_Code_Seq':'Num_of_visits'}, inplace = True)
visit_count.reset_index(drop=False)
visit_count.head()

#see how many records after aggregation
visit_count.info()

# 50738 unique patients exist
visit_count['enc_id'].unique


### Group similar records and order Visits
visits = cpt_codes.groupby('enc_id').apply(lambda x: x.sort_values('CPT_Code_Seq'))

visits = visits[['enc_id','CPT_Code']]
visits.reset_index(drop=True,level=0, inplace=True)
visits.head(10)

visits.info()

visits_sparse = pd.get_dummies(data = visits, columns=['CPT_Code'])
visits_sparse.head(10)

visits_sparse = visits_sparse.groupby(['enc_id']).agg("sum")
pd.set_option('display.max_columns', 500)
visits_sparse.reset_index(level=0, inplace=True)
visits_sparse.head(10)

# Verify 000462AF-CA71-4CDF-A9B3-84F354C78E1F has CPT codes 99213, 3075F, 3078F, 1126F, 1159F
visits_sparse[visits_sparse['enc_id'] == '000462AF-CA71-4CDF-A9B3-84F354C78E1F']

# Flattened data has same number of records as number of unique enc_id's
visits_sparse.info()
# %%
