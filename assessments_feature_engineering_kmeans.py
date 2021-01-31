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


# %% kmeans clustering for all assessments
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
tfidf = TfidfVectorizer()

tfidf_data_tokens = tfidf.fit_transform(assessment2['txt_tokenized2'])
tfidf_data_ngram = tfidf.fit_transform(assessment2['ngram2'])

kmeans = KMeans(init = "k-means++", n_jobs=-1, random_state = 42)
print("Starting token kmeans model fit...")
token_kmean_model = kmeans.fit(tfidf_data_tokens)
print("Token kmeans model fit COMPLETE...\n")
print("Starting ngram kmeans model fit...")
ngram_kmean_model = kmeans.fit(tfidf_data_ngram)
print("ngram kmeans model fit COMPLETE...")


assessment2['token_cluster'] = token_kmean_model.labels_
assessment2['ngram_cluster'] = ngram_kmean_model.labels_

# %% EDA on Clusters

from sklearn.cluster import DBSCAN

cluster_model = DBSCAN(n_jobs=-1)
ngram_db_model = cluster_model.fit(tfidf_data_ngram)
assessment2['ngram_cluster_db'] = ngram_db_model.labels_

print("Token Model Cluster Count:",assessment2['token_cluster'].nunique())
print("ngram Model Cluster Count:",assessment2['ngram_cluster'].nunique())
print("ngram DBSCAN Model Cluster Count:",assessment2['ngram_cluster_db'].nunique())

### Token model EDA #########################################################################

#Top10 tokens per cluster


### ngram model EDA #########################################################################

#Top10 trigrams per cluster

# %%
