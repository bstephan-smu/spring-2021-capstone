# %% Requirements:
import pandas as pd

data_path = 'E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/'

# Fix header on encounters:
new_header = ['person_id','enc_id','place_of_service','Provider_id','EncounterDate','Race','Ethnicity','Gender','AgeAtEnc','VisitType','ServiceDepartment','LocationName','Reason_for_Visit','CPT_Code','CPT_Code_Seq','Encounter_Primary_Payer','Encounter_Secondary_Payer','Encounter_Teritiary_Payer']
encounters = pd.read_csv(data_path + '1_BaseEncounters_Dempgraphics_Payers.csv', header=None, names=new_header, skiprows=1)

# Load other dataframes:
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

# Rename cols: 
assessments.columns = [
    'person_id', 
    'enc_id', 
    'txt_description', 
    'diagnosis_code_id',
    'txt_enc_dx_priority', 
    'detail_type', 
    'detail_type_priority'
    ]

diagnoses.columns = [
    'person_id', 
    'enc_id', 
    'icd9cm_code_id',
    'diagnosis_code_id',
    'description', 
    'date_onset_sympt', 
    'date_diagnosed', 
    'date_resolved',
    'status_id', 
    'dx_priority', 
    'chronic_ind', 
    'recorded_elsewhere_ind'
]

labs.columns = [
    'lab_nor_person_id', 
    'lab_nor_enc_id', 
    'lab_nor_order_num',
    'lab_nor_ordering_provider', 
    'lab_nor_test_location',
    'lab_nor_sign_off_date', 
    'lab_nor_test_status', 
    'lab_nor_ngn_status',
    'lab_nor_test_desc', 
    'lab_nor_delete_ind', 
    'lab_nor_completed_ind',
    'lab_results_obr_p_seq_num', 
    'lab_results_obr_p_obs_batt_id',
    'lab_results_obr_p_producer_field_1', 
    'lab_results_obr_p_test_desc',
    'lab_results_obr_p_ng_test_desc', 
    'lab_results_obr_p_coll_date_time',
    'lab_results_obr_p_spec_rcv_date_time', 
    'lab_results_obx_observ_value',
    'lab_results_obx_units', 
    'lab_results_obx_ref_range',
    'lab_results_obx_abnorm_flags', 
    'lab_results_obx_observ_result_stat',
    'lab_results_obx_obs_date_time', 
    'lab_results_obx_result_desc',
    'lab_results_obx_delete_ind', 
    'lab_results_obx_result_comment'
    ]

meds.columns = [
    'person_id', 
    'enc_id', 
    'ndc_id', 
    'start_date', 
    'date_stopped',
    'sig_codes', 
    'rx_quanity', 
    'rx_refills', 
    'generic_ok_ind',
    'org_refills', 
    'date_last_refilled', 
    'sig_desc', 
    'prescribed_else_ind',
    'rx_units', 
    'rx_comment', 
    'formulary_id', 
    'refills_left',
    'medid',
    'medication_name'
    ]

vitals.columns = [
    'person_id', 
    'enc_id', 
    'seq_no', 
    'age_cal', 
    'BMI_calc',
    'bp_body_position_2', 
    'bp_cuff_size', 
    'bp_diastolic', 
    'bp_systolic',
    'BP_target_side', 
    'BP_target_site', 
    'bp_target_site_cd', 
    'comments',
    'height_cm', 
    'height_in', 
    'height_date', 
    'pulse_pattern_2',
    'pulse_rate', 
    'respiration_rate', 
    'temp_deg_F', 
    'temp_targt_site_cd',
    'vitalSignsDate', 
    'vitalSignsTime', 
    'weight_context', 
    'weight_lb',
    'painScoreDisplay'
    ]

cpt_codes.columns = [
    'enc_id', 
    'CPT_Code', 
    'CPT_Code_Seq'
    ]

encounters.columns = [
    'person_id', 
    'enc_id',
    'Provider_id',
    'EncounterDate', 
    'Race',
    'Ethnicity', 
    'Gender',
    'AgeAtEnc', 
    'VisitType', 
    'ServiceDepartment',
    'LocationName', 
    'Reason_for_Visit',
    'CPT_Code',
    'Encounter_Primary_Payer',
    'Encounter_Secondary_Payer',
    'Encounter_Teritiary_Payer'
    ]    


# %%
#assessments.pivot(index=['person_id','enc_id','diagnosis_code_id'], columns = 'detail_type', values = 'txt_description')

# Find dupes:

assessments.groupby(['person_id','enc_id','diagnosis_code_id','detail_type']).count().sort_values(by='txt_description', ascending=False)


# %%
# Examine dupes:
assessments[assessments.enc_id == 'B4A43273-1B47-4B68-BC27-74EE22ECE620'].sort_values(by='diagnosis_code_id').to_csv('dupes.csv')


# %% NOT RUN, code doesn't work yet due to dupes...

#assessments = pd.concat([
#    assessments['person_id','enc_id','diagnosis_code_id'],
#    assessments.pivot(columns = 'detail_type', values = 'txt_description')
#    ], axis=1)

#assessments = assessments.drop(columns=['detail_type','detail_type_priority', 'txt_enc_dx_priority', 'txt_description']).drop_duplicates()

# %%
tmp = assessments.merge(assessments, how='inner', on=['person_id','enc_id','diagnosis_code_id'] 
tmp


# %% MERGE Section
# begin merge
df = assessments.merge(
    diagnoses, 
    how='left', 
    on=['person_id','enc_id','diagnosis_code_id']
)
# %%
len(assessments)

# %%
len(df)
# %%
tmp = assessments[assessments.enc_id == '356C735C-8F94-464A-88F6-24976C094EFD'].sort_values(by='diagnosis_code_id')
# %%
