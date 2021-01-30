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

#%% Isolate AD and dementia patients from diagnoses table
diagnoses = pd.read_csv(data_path + '6_patient_diagnoses.csv')

ngd_icd10 = 'F01|F02|F03|F10|F32|F68|G20|G30|G31|G91|Q05|S09|Z63|Z82|Z81'

ngd = diagnoses[diagnoses['icd9cm_code_id'].str.contains(ngd_icd10,case = False)]
diagnoses['icd9cm_code_id'].str.contains(ngd_icd10,case = False).value_counts()

ngd.reset_index(drop=True)
ngd.head()

#%% Identify unique records with neruogentive diseases
ngd.nunique()

#%% Merge assements[txt_description] to ngd df
ngd_txt = ngd.merge(assessment2, how = 'left', on = ['person_id','enc_id'])


#%% Take a look at ngd_text df
ngd_txt.head(15)
ngd_txt.shape


#%% Remove punctuation, convert all to lowercase, remove stopwords, tokenize, create bigrams
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

# Remove Punctuation and convert to lower
ngd_txt['txt_description'] = ngd_txt.txt_description.apply(lambda x: ', '.join([str(i) for i in x]))
ngd_txt['txt_description'] = ngd_txt['txt_description'].str.replace('[^\w\s]','')
ngd_txt['txt_description'] = ngd_txt['txt_description'].str.lower()

#tokenize
ngd_txt['txt_tokenized']= ngd_txt.apply(lambda row: nltk.word_tokenize(row['txt_description']), axis=1)

#Remove Stopwords
stop = stopwords.words('english')
ngd_txt['txt_tokenized'] =ngd_txt['txt_tokenized'].apply(lambda x: [item for item in x if item not in stop])

#Create bigrams
ngd_txt['bigrams'] = ngd_txt.apply(lambda row: list(nltk.trigrams(row['txt_tokenized'])),axis=1) 
#print(*map(' '.join, bigrm), sep=', ')

ngd_txt.head()

#%% Frequent and common terms
pd.set_option('display.max_rows', 100)
ngd_txt['bigram2'] = ngd_txt.bigrams.apply(lambda row:['_'.join(i) for i in row])
ngd_txt.head()


# %% Convert trigram lists to strings
ngd_txt['bigram2'] = ngd_txt.bigram2.apply(lambda x: ' '.join([str(i) for i in x]))
ngd_txt.head()



# %% create table to display trigram counts
word_counts = pd.Series(' '.join(ngd_txt.bigram2).split()).value_counts()
word_counts = word_counts.reset_index()
word_counts.columns = ['trigram','count']
word_counts = pd.DataFrame(word_counts)
word_counts.head(100)


# %% generate word cloud from trigrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt 

#convert word_count df to dictionary for ease of use in wordcloud()
word_counts_dict = dict(zip(word_counts['trigram'].tolist(),word_counts['count'].tolist()))

#make the wordcloud
cloud = WordCloud(width = 1600, height = 800, max_font_size=100,max_words = 100, background_color = 'white', relative_scaling = 1, colormap ="viridis").generate_from_frequencies(word_counts_dict)
plt.figure(figsize=(20,10))
plt.imshow(cloud)
plt.axis('off')
plt.show()

# %% Topic modeling
