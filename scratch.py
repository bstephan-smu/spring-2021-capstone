#%% Requirements:
import numpy as np
from numpy.lib.histograms import histogram
import pandas as pd

# %% Load data:
data_path = 'E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/'

# Fix header on encounters:
new_header = ['person_id','enc_id','place_of_service','Provider_id','EncounterDate','Race','Ethnicity','Gender','AgeAtEnc','VisitType','ServiceDepartment','LocationName','Reason_for_Visit','CPT_Code','CPT_Code_Seq','Encounter_Primary_Payer','Encounter_Secondary_Payer','Encounter_Teritiary_Payer']
encounters = pd.read_csv(data_path + '1_BaseEncounters_Dempgraphics_Payers.csv', header=None, names=new_header, skiprows=1)

cpt_codes = pd.read_csv(data_path + '2_CPT_Codes.csv')
vitals = pd.read_csv(data_path + '3_vitals_signs.csv')
meds = pd.read_csv(data_path + '4_patient_medication.csv')
labs = pd.read_csv(data_path + '5_lab_nor__lab_results_obr_p__lab_results_obx.csv')
diagnoses = pd.read_csv(data_path + '6_patient_diagnoses.csv')
assessments = pd.read_csv(data_path + '7_assessment_impression_plan_.csv')

# Drop columns with only one value, or contain all NAs:
encounters = encounters.drop(columns=['place_of_service', 'CPT_Code_Seq'])
vitals = vitals.drop(columns=['age_indicator','bp_comment','pulse_pattern','BP','bp_body_position'])

# Collect response
AD_people = diagnoses[diagnoses.description.str.contains('Alzh')].person_id.unique()
AD_encounters = diagnoses[diagnoses.description.str.contains('Alzh')].enc_id.unique()

##### EDA SECTION #####

# %% check NAs:
def NA_dist(df):
    s = df.isna().sum()
    d = df.isnull().sum()/len(df)*100
    result = pd.concat([s,d], axis=1)
    result.columns = ['NA_Count','NA_Percent']
    return result

dfs = [encounters, cpt_codes, vitals, meds, labs, diagnoses, assessments]

for df in dfs:
    print(NA_dist(df))



# %% examine response:

# Unique people with AD: 517
len(AD_people) 

# Unique encounters with AD: 2252
len(AD_encounters)

# Unique people without AD: 4995
len(diagnoses[~diagnoses.description.str.contains('Alzh')].person_id.unique())


# %% Check distribution of encounter dates:
e = encounters.EncounterDate.sort_values().unique()
e.tofile('dates.csv', sep='\n')


# %%
