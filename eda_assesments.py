#%% Import the data

import pandas as pd
data_path = 'E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/'
assessment = pd.read_csv(data_path + '7_assessment_impression_plan_.csv')

# %% Take a look at the initial data
assessment.head()

#%% Get the shape of the original data
assessment.shape
assessment['person_id'].nunique()

#%% Identify number of duplicates
assessment.duplicated(subset=['person_id','enc_id']).sum()


#%% Collapse data down by enc_id

assessment_text = assessment.groupby(['person_id','enc_id'])['txt_description'].apply(list)
assessment_codeID = assessment.groupby(['person_id','enc_id'])['txt_diagnosis_code_id'].apply(list)

assessment_text = pd.DataFrame(assessment_text)
assessment_codeID = pd.DataFrame(assessment_codeID)

#%% Merge series data from text and codeID columns into one df
assessment2 = assessment_text.merge(assessment_codeID, how = 'left', on = ['person_id','enc_id'])
assessment2 = pd.DataFrame(assessment2)
assessment2.reset_index(inplace=True)

#%% Convert txt_description list to string for easier processing
assessment2['txt_description'] = assessment2.txt_description.apply(lambda x: ', '.join([str(i) for i in x]))

#%% Take a look at the data
assessment2.head(20)

# %% Check Shape of Data, reduction from 1,545,049 records to 208,400 records
assessment2.shape

# %% Check for dup enc_id, result is 0
assessment2['enc_id'].duplicated().sum()

# %% check number of unique patient_ids
assessment2['person_id'].nunique()
