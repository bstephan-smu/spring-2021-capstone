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
from os import getcwd

from nltk.tokenize import word_tokenize
from load_data import DataLoader
from numpy.core.numeric import NaN
from numpy.lib.arraysetops import unique
from pandas.core import groupby
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import union_indexes
from plotnine import *

ggplot(encounters) + geom_bar(aes(x='Race', fill='Cognition')) + coord_flip()

#ggplot(encounters) + geom_bar(aes(x='Ethnicity', fill='Cognition')) + coord_flip()



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

capData.main.diagnosis_code_id

# %%
capData.main.to_csv(capData.data_path+'main.csv')
# %%
# Pandas get_dummies function will not parse lists, enter the multiLabelBinarizer 

def one_hot(df, col_name, prefix=''):
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    df = df.join(pd.DataFrame(mlb.fit_transform(df[col_name]),columns=prefix+mlb.classes_))
    df = df.drop(columns=[col_name])
    return df

tmp = one_hot(capData.asmt_diag, 'diagnosis_code_id', prefix='icd_')
#tmp.fillna(0, inplace=True)
tmp


# %%
capData.asmt_diag[np.isnan(capData.asmt_diag['diagnosis_code_id'])]


# %%
import numpy as np
import pandas as pd
df = mlb.fit_transform(capData.main.asmt_diagnosis_code_id)
columns='icd_'+mlb.classes_

pd.DataFrame(df, columns=columns)

# %%
tmp[tmp['icd_Z98.89'] >0]
# %% 
def NA_dist(df):
    s = df.isna().sum()
    d = df.isnull().sum()/len(df)*100
    result = pd.concat([s,d], axis=1)
    result.columns = ['NA_Count','NA_Percent']
    return result

import pandas as pd
NA_dist(capData.main).sort_values(by='NA_Percent')



# %% dianosis

import re
def find_diag(lookup, return_val='description'):
    print('Terms = ', lookup) 
    result = list(capData.diagnosis[capData.diagnosis.description.str.contains(lookup, regex=True, flags=re.IGNORECASE)][return_val].unique())
    return result
# %%

diag_codes = (code.lstrip('asmt_icd_') for code in iDF.nlargest(20, 'Feature_Importance').Feature)
descriptions = find_diag('|'.join(diag_codes))

print(descriptions)
list(zip(diag_codes, descriptions))


# %% Meds exploration

df = capData.meds

set(sorted(df['medication_name'].str.lower()))


# %%

x = 'ferrous sulfate 324 mg (65 mg iron) tablet,delayed release'
#x = 'Co Q-10 300 mg capsule'
#x = 'Detrol 2 mg tablet'
#x = 'Doc-Q-Lax 8.6 mg-50 mg tablet'

import re
y = re.search('(\\D)(.*)',x, flags=re.IGNORECASE).groups()
y

# %%


sorted(df['short_name'].unique())
# %%
df['med'].nunique() #2784
df['medid'].nunique() #4547
df['medication_name'].nunique() #5151

# %%
sorted(df['medication_name'].unique())

# %%
def get_med_name(med):
    med = re.search('(\\D+)(.*)',med, flags=re.IGNORECASE).group(1).strip().lower()
    if med.endswith('-'):
        med = med[:-1]
    return med


# %%

capData.meds['med'] = capData.meds['medication_name'].apply(get_med_name)



# %%
medcols_old = [col for col in capData.main if col.startswith('med')]

df = capData.meds[['medid', 'med']]
dic = df.to_dict()

medcols_new = []

for col in medcols_old:
    id = col.split('medid_')[1]
    print(dic.get(id))

# %%
df[df['medid']=='825']
# %%
df
# %%
