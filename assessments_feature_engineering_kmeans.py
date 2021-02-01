#%% Import the data

import pandas as pd
data_path = 'E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/'
assessment = pd.read_csv(data_path + '7_assessment_impression_plan_.csv')

# %% Take a look at the initial assessment data
assessment.head()

#%% Get the shape of the original assessment data
assessment.shape
assessment['person_id'].nunique()

#%% Identify number of duplicates in assessment
assessment.duplicated(subset=['person_id','enc_id']).sum()


#%% Collapse data down by enc_id for assessment

assessment_text = assessment.groupby(['person_id','enc_id'])['txt_description'].apply(list)
assessment_codeID = assessment.groupby(['person_id','enc_id'])['txt_diagnosis_code_id'].apply(list)

assessment_text = pd.DataFrame(assessment_text)
assessment_codeID = pd.DataFrame(assessment_codeID)

#%% Merge series data from text and codeID columns into one df for assessment
assessment2 = assessment_text.merge(assessment_codeID, how = 'left', on = ['person_id','enc_id'])
assessment2 = pd.DataFrame(assessment2)
assessment2.reset_index(inplace=True)

#%% Take a look at the data aggregated assessment data
assessment2.head(20)

#%% Check Shape of assessment2 Data, reduction from 1,545,049 records to 208,400 records
assessment2.shape

#%% Check for dup enc_id, result is 0
assessment2['enc_id'].duplicated().sum()

#%% check number of unique patient_ids
assessment2['person_id'].nunique()

#%% Remove punctuation, convert all to lowercase, remove stopwords, tokenize, create bigrams
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

# Remove Punctuation and convert to lower
assessment2['txt_description'] = assessment2.txt_description.apply(lambda x: ', '.join([str(i) for i in x]))
assessment2['txt_description'] = assessment2['txt_description'].str.replace('[^\w\s]','')
assessment2['txt_description'] = assessment2['txt_description'].str.lower()

#tokenize
assessment2['txt_tokenized']= assessment2.apply(lambda row: nltk.word_tokenize(row['txt_description']), axis=1)

#Remove Stopwords
stop = stopwords.words('english')
assessment2['txt_tokenized'] =assessment2['txt_tokenized'].apply(lambda x: [item for item in x if item not in stop])

#Create ngrams
assessment2['ngrams'] = assessment2.apply(lambda row: list(nltk.trigrams(row['txt_tokenized'])),axis=1) 

assessment2.head()


#%% Convert lists to strings
pd.set_option('display.max_rows', 100)

# Convert trigram lists to words joined by underscores
assessment2['ngram2'] = assessment2.ngrams.apply(lambda row:['_'.join(i) for i in row])

# Convert trigram and token lists to strings
assessment2['txt_tokenized2'] = assessment2['txt_tokenized'].apply(' '.join)
assessment2['ngram2'] = assessment2.ngram2.apply(lambda x: ' '.join([str(i) for i in x]))

assessment2.head()

#%% Get noun phrases
import spacy
import en_core_web_sm

nlp = en_core_web_sm.load()

def getNounChunks(text_data):
    doc = nlp(text_data)
    noun_chunks = list(doc.noun_chunks)
    noun_chunks_strlist = [chunk.text for chunk in noun_chunks]
    noun_chunks_str = '_'.join(noun_chunks_strlist)
    return noun_chunks_str

assessment2['np_chunks'] = assessment2['txt_tokenized2'].apply(getNounChunks)
assessment2.head()

#%% Checdk unqiue number of patient ids in assessments table
assessment2['person_id'].nunique()

#%% Pair down assessments table to columns of interest
assessment2 = assessment2[['person_id','enc_id','txt_description','txt_tokenized','ngrams','ngram2','txt_tokenized2','np_chunks']]
assessment2.head()

#%% DBSCAN Clusterin for trigrams and noun phrase chunks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

tfidf = TfidfVectorizer()

tfidf_data_ngram = tfidf.fit_transform(assessment2['ngram2'])
tfidf_data_np = tfidf.fit_transform(assessment2['np_chunks'])

cluster_model = KMeans(n_jobs=-1,n_clusters=15)
print("Starting ngram DBSCAN model fit...")
ngram_db_model = cluster_model.fit(tfidf_data_ngram)
print("ngram DBSCAN model fit COMPLETE...")

print("Starting ngram DBSCAN model fit on np chunks...")
np_db_model = cluster_model.fit(tfidf_data_np)
print("ngram DBSCAN model fit on np chunks COMPLETE...")


#%% Get cluster counts
assessment2['ngram_clusters'] = ngram_db_model.labels_
assessment2['np_chunk_clusters'] = np_db_model.labels_

print("ngram Model Cluster Count:",assessment2['ngram_clusters'].nunique())
print("ngram DBSCAN Model Cluster Count:",assessment2['np_chunk_clusters'].nunique())


#%% Read in diagnosis table
diagnoses = pd.read_csv(data_path + '6_patient_diagnoses.csv')

diagnosis_icd9 = pd.DataFrame(diagnoses.groupby(['person_id','enc_id'])['icd9cm_code_id'].apply(list))
diagnosis_dc = pd.DataFrame(diagnoses.groupby(['person_id','enc_id'])['diagnosis_code_id'].apply(list))
diagnosis_desc = pd.DataFrame(diagnoses.groupby(['person_id','enc_id'])['description'].apply(list))
diagnosis_datesymp = pd.DataFrame(diagnoses.groupby(['person_id','enc_id'])['date_onset_sympt'].apply(list))
diagnosis_datediag = pd.DataFrame(diagnoses.groupby(['person_id','enc_id'])['date_diagnosed'].apply(list))
diagnosis_dateresl = pd.DataFrame(diagnoses.groupby(['person_id','enc_id'])['date_resolved'].apply(list))
diagnosis_statusid = pd.DataFrame(diagnoses.groupby(['person_id','enc_id'])['status_id'].apply(list))
diagnosis_dx = pd.DataFrame(diagnoses.groupby(['person_id','enc_id'])['dx_priority'].apply(list))
diagnosis_chronic = pd.DataFrame(diagnoses.groupby(['person_id','enc_id'])['chronic_ind'].apply(list))
diagnosis_rcdelswhr = pd.DataFrame(diagnoses.groupby(['person_id','enc_id'])['recorded_elsewhere_ind'].apply(list))


#%% Merge series data from text and codeID columns into one df for assessment


diagnoses2 = diagnosis_icd9.merge(diagnosis_dc, how = 'left', on = ['person_id','enc_id']).merge(diagnosis_desc, how = 'left', on = ['person_id','enc_id']).merge(diagnosis_datesymp, how = 'left', on = ['person_id','enc_id']).merge(diagnosis_datediag, how = 'left', on = ['person_id','enc_id']).merge(diagnosis_dateresl, how = 'left', on = ['person_id','enc_id']).merge(diagnosis_statusid, how = 'left', on = ['person_id','enc_id']).merge(diagnosis_dx, how = 'left', on = ['person_id','enc_id']).merge(diagnosis_chronic, how = 'left', on = ['person_id','enc_id']).merge(diagnosis_rcdelswhr, how = 'left', on = ['person_id','enc_id'])
diagnoses2 = pd.DataFrame(diagnoses2)
diagnoses2.reset_index(inplace=True)

#%% Merge assements[txt_description] to ngd df
#Diagnosis occur after assessments, diagnosis table is smaller than assessments table
assessments_diagnoses = assessment2.merge(diagnoses2, how = 'left', on = ['person_id','enc_id'])
assessments_diagnoses.head()

#%% Write to CSV
assessments_diagnoses.to_csv("assessments_diagnoses_join.csv")

