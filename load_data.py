# %% Requirements:
import pandas as pd
import re

# Load data:
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


### SECTION Response variables: ###
def find_diag(lookup, return_val='description'):
    print('Terms = ', lookup) 
    result = list(diagnoses[diagnoses.description.str.contains(lookup, regex=True, flags=re.IGNORECASE)][return_val].unique())
    return result

# Dementia indicators:
dementia_lookup = [
    'alzh',
    'lewy',
    'dementia',
    'frontotemporal',
    'hydrocephalus',
    'huntington',
    'wernicke',
    'creutzfeldt'
]

dem_string = '|'.join(dementia_lookup)
dem_descriptions = find_diag(dem_string)

# Remove non-dementia diagnosis codes:
non_dementia = [
    'Family history of dementia',
    "Family history of Alzheimer's disease",
    'Stress due to spouse with dementia',
    'Sacral spina bifida without hydrocephalus',
    'Spina bifida, unspecified hydrocephalus presence, unspecified spinal region'
    ]
dem_descriptions = [desc for desc in dem_descriptions if not desc in non_dementia]

# Output dementia ICD codes
dementia_ICD_codes = diagnoses[diagnoses.description.isin(dem_descriptions)].icd9cm_code_id.unique()

# Collect response
AD_people = diagnoses[diagnoses.description.str.contains('Alzh')].person_id.unique()
AD_encounters = diagnoses[diagnoses.description.str.contains('Alzh')].enc_id.unique()
dem_people = diagnoses[diagnoses.description.isin(dem_descriptions)].person_id.unique()
dem_encounters = diagnoses[diagnoses.description.isin(dem_descriptions)].enc_id.unique()

# Set response
encounters['AD_event'] = encounters.enc_id.isin(AD_encounters).astype(int)
encounters['AD_person'] = encounters.person_id.isin(AD_people).astype(int)
encounters['dem_event'] = encounters.enc_id.isin(dem_encounters).astype(int)
encounters['dem_person'] = encounters.person_id.isin(dem_people).astype(int)





# %%
