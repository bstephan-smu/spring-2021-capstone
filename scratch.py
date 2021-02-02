#%% load_data:

# Run load data.py first
pd.set_option('display.max_columns', None)


##### EDA SECTION #####


## Count stuff: 

# Unique people with AD: 517
len(AD_people) 

# Unique events with AD: 2252
len(AD_encounters)

# Unique people without AD: 4995
len(diagnoses[~diagnoses.description.str.contains('Alzh')].person_id.unique())

# Check distribution of encounter dates: looks normal
encounters.EncounterDate.sort_values().unique()

# Count dementia descriptions & codes:
len(find_diag('|'.join(dementia_lookup))) #118 unique descriptions
len(find_diag('|'.join(dementia_lookup), 'icd9cm_code_id')) #25 unique codes


## List Stuff:
print('Lewy')
list(diagnoses[diagnoses.description.str.contains('lewy', regex=True, flags=re.IGNORECASE)].description.unique())

print('alzh')
list(diagnoses[diagnoses.description.str.contains('alzh', regex=True, flags=re.IGNORECASE)].description.unique())


# %% check NAs:
def NA_dist(df):
    s = df.isna().sum()
    d = df.isnull().sum()/len(df)*100
    result = pd.concat([s,d], axis=1)
    result.columns = ['NA_Count','NA_Percent']
    return result

dfs = {
    'encounters':encounters, 
    'cpt_codes':cpt_codes, 
    'vitals':vitals, 
    'meds':meds, 
    'labs':labs, 
    'diagnoses':diagnoses, 
    'assessments':assessments
}

for df in dfs:
    print(df.center(60, '-'), '\n', NA_dist(dfs[df]),'\n')


# %% Plots
from load_data import DataLoader
from numpy.core.numeric import NaN
from numpy.lib.arraysetops import unique
from pandas.core import groupby
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import union_indexes
from plotnine import *

ggplot(encounters) + geom_bar(aes(x='Race', fill='Cognition')) + coord_flip()

#ggplot(encounters) + geom_bar(aes(x='Ethnicity', fill='Cognition')) + coord_flip()






# %% ETL codes:  NOT RUN!!
# cpts = pd.read_csv('cpt_updated_full.csv')
# cpts.to_csv('cpt_descriptions.csv', index=False)

# icds = pd.read_csv('icd_updated.csv')
# icds = icds.drop(columns=['Unnamed: 0', 'id','Freq','VALIDITY_x','STATUS_x','CODE_TYPE_x'])
# icds.columns =  ['icd', 'short_description', 'long_description', 'full_description']
# icds.to_csv('icd_descriptions.csv', index=False)


# %% ETL


# Drop columns with only one value, or contain all NAs:
encounters = encounters.drop(columns=['place_of_service', 'CPT_Code_Seq'])
vitals = vitals.drop(columns=['age_indicator','bp_comment','pulse_pattern','BP','bp_body_position'])


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
tmp = assessments.merge(assessments, how='inner', on=['person_id','enc_id','diagnosis_code_id'] )
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



# %% lab section
[print(u, ' - ', labs[u].nunique()) for u in labs]
# %%
# Remove Deleted rows and drop deleted indicators
labs = labs[labs['lab_nor_delete_ind']=='N']
labs = labs.drop(columns=['lab_nor_delete_ind', 'lab_results_obx_delete_ind'])

# Remove incomplete labs & drop column
labs = labs[labs['lab_nor_completed_ind']=='Y']
labs = labs.drop(columns=['lab_nor_completed_ind'])

# Remove pending labs
labs = labs[labs['lab_nor_test_status']!='InProcessUnspecified']
labs = labs[labs['lab_nor_test_status']!='Pending']



# %%
labs[labs['lab_nor_person_id'].isin(set(labs['lab_nor_person_id']).intersection(set(encounters.person_id)))]
labs[labs['lab_nor_enc_id'].isin(set(labs['lab_nor_enc_id']).intersection(set(encounters.enc_id)))]

# %%
labs.lab_nor_ordering_provider.unique()

# %%
labs.sort_values(by='lab_nor_enc_id')

# %%
labs.lab_nor_test_desc.nunique()
# %%
labs[labs['lab_nor_order_num']=='A283DA59-D234-4D33-8FE8-80F85A5CCD37']
# %%
labs.lab_nor_test_status.unique()
# %%
# Find abnormal tests:
abnormal_indicators = ['L', 'H', 'A', '>', 'LL', 'HH', '<']
labs[labs['lab_results_obx_abnorm_flags'].isin(abnormal_indicators)]
# %%
labs['lab_results_obx_abnorm_flags'].unique()
# %%

labs['Abnormal_Test'] = np.where(
    labs['lab_results_obx_abnorm_flags'].isin(abnormal_indicators),
    labs['lab_results_obr_p_test_desc'],
    np.nan
)

# %%
labs.lab_results_obx_result_desc.nunique()
# %%
labs.lab_results_obr_p_test_desc.nunique()
# %%
labs.lab_results_obr_p_ng_test_desc.nunique()


# %%
labs.lab_nor_test_desc.nunique()
# %%
labs[labs['Abnormal_Test'].notnull()]
# %%
labs[labs['lab_results_obx_abnorm_flags']=='LL']
# %%

# %%
labs['lab_results'] = np.where(
    labs['lab_results_obx_result_desc'].notnull(), 
    labs['lab_results_obx_result_desc'], 
    labs['lab_results_obr_p_test_desc']
    )


labs['lab_results'] = np.select(
    [ #condition
        labs.lab_results_obx_abnorm_flags == '>',
        labs.lab_results_obx_abnorm_flags == 'H',
        labs.lab_results_obx_abnorm_flags == 'HH', 
        labs.lab_results_obx_abnorm_flags == '<', 
        labs.lab_results_obx_abnorm_flags == 'L', 
        labs.lab_results_obx_abnorm_flags == 'LL', 
        labs.lab_results_obx_abnorm_flags == 'A'        
        ],
    [ #value
        'HIGH ' + labs['lab_results'],
        'HIGH ' + labs['lab_results'],
        'VERY HIGH ' + labs['lab_results'],
        'LOW ' + labs['lab_results'],
        'LOW ' + labs['lab_results'],
        'VERY LOW ' + labs['lab_results'],
        'ABNORMAL ' +labs['lab_results']
        ],
    default = 'NORMAL'
)

labs[labs['lab_results'].notnull()]

# %%
abnormal_labs = labs[labs['lab_results'] != 'NORMAL']
abnormal_labs = pd.DataFrame(abnormal_labs[['lab_nor_person_id','lab_nor_enc_id','lab_results']].groupby(
    ['lab_nor_person_id','lab_nor_enc_id'])['lab_results'].apply(set))
    
abnormal_labs.reset_index(inplace=True)    
abnormal_labs


# %%
pd.get_dummies(abnormal_labs, columns='lab_results')
# %%
encounters.groupby('enc_id').agg({'CPT_Code':'count'}).sort_values(by='CPT_Code')

#list(encounters.columns)
# %%
cpt_codes.groupby('enc_id').agg({'CPT_Code':'count'}).sort_values(by='CPT_Code')
diagnoses.groupby('enc_id').agg({'diagnosis_code_id':'count'}).sort_values(by='diagnosis_code_id')



# %%
from load_data import DataLoader
capData = DataLoader(subset=10)
capData.generate_csv_attributes()
capData.encode_alzheimers()


# %%
 
assessments = capData.assessments

def format_assessment():
        assessment_text = pd.DataFrame(assessments.groupby(['person_id', 'enc_id'])['txt_description'].apply(list))
        assessment_codeID = pd.DataFrame(assessments.groupby(['person_id', 'enc_id'])['txt_diagnosis_code_id'].apply(list))

        # %% Merge series data from text and codeID columns into one df for assessment
        assessment2 = assessment_text.merge(assessment_codeID, how='left', on=['person_id', 'enc_id'])
        assessment2 = pd.DataFrame(assessment2)
        assessment2.reset_index(inplace=True)

        # Remove Punctuation and convert to lower
        assessment2['txt_description'] = assessment2.txt_description.apply(lambda x: ', '.join([str(i) for i in x]))
        assessment2['txt_description'] = assessment2['txt_description'].str.replace('[^\w\s]', '')
        assessment2['txt_description'] = assessment2['txt_description'].str.lower()

        # tokenize
        assessment2['txt_tokenized'] = assessment2.apply(lambda row: nltk.word_tokenize(row['txt_description']), axis=1)

        # Remove Stopwords
        stop = stopwords.words('english')
        assessment2['txt_tokenized'] = assessment2['txt_tokenized'].apply(
            lambda x: [item for item in x if item not in stop])

        # Create ngrams
        assessment2['ngrams'] = assessment2.apply(lambda row: list(nltk.trigrams(row['txt_tokenized'])), axis=1)
        # Convert trigram lists to words joined by underscores
        assessment2['ngram2'] = assessment2.ngrams.apply(lambda row: ['_'.join(i) for i in row])

        # Convert trigram and token lists to strings
        assessment2['txt_tokenized2'] = assessment2['txt_tokenized'].apply(' '.join)
        assessment2['ngram2'] = assessment2.ngram2.apply(lambda x: ' '.join([str(i) for i in x]))

        # %% Pair down assessments table to columns of interest
        assessment2 = assessment2[
            ['person_id', 'enc_id', 'txt_description', 'txt_tokenized', 'ngrams', 'ngram2', 'txt_tokenized2']]

        return assessment2
# %%
import pandas as pd
import nltk
from nltk.corpus import stopwords
format_assessment()


# %%
from load_data import DataLoader
capData = DataLoader()
capData.create()


# %%

assessments = pd.read_csv(data_path + '7_assessment_impression_plan_.csv')
encounters = pd.read_csv(data_path + '1_BaseEncounters_Dempgraphics_Payers.csv')
assessments[assessments['enc_id'].isin(
    set(assessments['enc_id']).intersection(
        set(encounters.enc_id)))].enc_id.nunique()



# %%
set(encounters.enc_id)
# %%
pd.DataFrame(encounters.enc_id.unique()).to_csv('encounters.csv')

# %%
datapath='E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/'
encounters = pd.read_csv(datapath + '1_BaseEncounters_Dempgraphics_Payers.csv')
merged_assessments = pd.read_csv(datapath+'assessments_diagnoses_join.csv')
merged_assessments = merged_assessments[merged_assessments['enc_id'].isin(
    set(merged_assessments['enc_id']).intersection(
        set(encounters.enc_id)))]


# %%
merged_assessments
# %%
from load_data import DataLoader
capData = DataLoader()
capData.create()

# %%
