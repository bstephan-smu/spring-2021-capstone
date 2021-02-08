# %% Requirements:
import pandas as pd
import numpy as np
import re
import os
import pickle
import nltk
import spacy
import en_core_web_sm

from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

#os.environ["MODIN_ENGINE"] = "dask"  # Modin will use Dask
#import modin.pandas as pd

class DataLoader:
    def __init__(self, data_path='E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/',
                 subset=None):
        # Load data: storing as environment variable in case sharing across multiple files
        os.environ['data_path'] = data_path
        self.data_path = data_path

        # can specify subset of data to work with...based on encounters table
        self.subset = subset

        # regex phrase to lookup alzheimers diagnosis
        self.alz_regex = 'alzh'

        # store list of dementia and AD icd codes here... TODO will be updated in later function
        self.dementia_icd_codes = None
        self.AD_icd_codes = None

        # list of common word codes for dementia diagnosis
        self.dementia_lookup = [
            'alzh',
            'lewy',
            'dementia',
            'frontotemporal',
            'hydrocephalus',
            'huntington',
            'wernicke',
            'creutzfeldt'
        ]

        # list of diagnosis decriptions to override and exclude from output
        self.exclude_dementia_lookup = [
            'Family history of dementia',
            "Family history of Alzheimer's disease",
            'Stress due to spouse with dementia',
            'Sacral spina bifida without hydrocephalus',
            'Spina bifida, unspecified hydrocephalus presence, unspecified spinal region'
        ]

        # initializing tables as attributes
        # to read in and process data, try self.generate_csv_attributes()
        self.encounters = None
        self.cpt = None
        self.vitals = None
        self.meds = None
        self.labs = None
        self.diagnosis = None
        self.assessments = None
        self.main = None

    # Fix header on encounters:
    def format_encounters(self):
        self.align_columns()  # ensure that encounters columns are properly aligned
        self.encounters['EncounterDate'] = pd.to_datetime(self.encounters['EncounterDate'], format='%Y%m%d')

        dropcols = ['Encounter_Primary_Payer', 'Encounter_Secondary_Payer', 'Encounter_Teritiary_Payer', 
                    'LocationName', 'ServiceDepartment', 'VisitType', 'CPT_Code', 'CPT_Code_Seq']
        self.encounters.drop(columns=dropcols, inplace=True)

        # in event that you'd like to subset less than total appt in data
        if self.subset is not None:
            self.encounters = self.encounters.head(
                self.subset)  # making a smaller sample of data for speed in dev environment

    # function to correct column alignment in the encounters table
    def align_columns(self):
        self.encounters.drop(columns=['Encounter_Teritiary_Payer'], inplace=True)
        self.encounters.columns = ['person_id', 'enc_id', 'place_of_service', 'Provider_id', 'EncounterDate', 'Race',
                                   'Ethnicity', 'Gender', 'AgeAtEnc', 'VisitType', 'ServiceDepartment',
                                   'LocationName',
                                   'Reason_for_Visit', 'CPT_Code', 'CPT_Code_Seq', 'Encounter_Primary_payer',
                                   'Encounter_Secondary_Payer', 'Encounter_Teritiary_Payer']

    # function to classify as True/False Alzheimers Disease in the encounter dataset
    # will also encode a separate dementia encoding.
    def encode_alzheimers(self, return_val='description'):
        dementia_string = '|'.join(self.dementia_lookup)
        dementia_output = list(
            self.diagnosis[self.diagnosis.description.str.contains(dementia_string, regex=True, flags=re.IGNORECASE)]
            [return_val].unique()
        )
        dementia_output = [desc for desc in dementia_output if desc not in self.exclude_dementia_lookup]
        self.dementia_icd_codes = self.diagnosis[
            self.diagnosis.description.isin(dementia_output)].icd9cm_code_id.unique()

        #TODO add self.AD_icd_codes

        # Collect response
        AD_people = self.diagnosis[
            self.diagnosis.description.str.contains(self.alz_regex, regex=True, flags=re.IGNORECASE)].person_id.unique()
        AD_encounters = self.diagnosis[
            self.diagnosis.description.str.contains(self.alz_regex, regex=True, flags=re.IGNORECASE)].enc_id.unique()
        dem_people = self.diagnosis[self.diagnosis.description.isin(dementia_output)].person_id.unique()
        dem_encounters = self.diagnosis[self.diagnosis.description.isin(dementia_output)].enc_id.unique()

        # Set response
        self.encounters['AD_event'] = self.encounters.enc_id.isin(AD_encounters).astype(int)
        self.encounters['AD_person'] = self.encounters.person_id.isin(AD_people).astype(int)
        self.encounters['dem_event'] = self.encounters.enc_id.isin(dem_encounters).astype(int)
        self.encounters['dem_person'] = self.encounters.person_id.isin(dem_people).astype(int)
        self.encounters['Cognition'] = np.select(
            [self.encounters.AD_person == 1, self.encounters.dem_person == 1],
            ['AD', 'Dementia'],
            default='Normal'
        )

        # Store in mainDF
        self.main = self.encounters.copy()


    def merge_cpt(self):
        df_cpt_codes_encoded = pd.concat(
            [
                self.cpt[['enc_id']], 
                pd.get_dummies(self.cpt['CPT_Code'], drop_first=True, prefix='cpt')
            ], axis=1) \
        .groupby('enc_id', as_index=False).max()

        self.main = self.main.merge(df_cpt_codes_encoded, on='enc_id')


    def merge_vitals(self, rename=True):
        # get average vital measurement per patient encounter
        vitals_agg = self.vitals[['enc_id', 'BMI_calc', 'bp_diastolic', 'bp_systolic',
                                  'height_cm', 'pulse_rate', 'respiration_rate', 'temp_deg_F',
                                  'weight_lb']].groupby('enc_id', as_index=False).max()

        vitals_copy = vitals_agg.copy()
        if rename:
            vitals_copy = self.rename_cols(vitals_copy, prefix='vit_')
        self.main = self.main.merge(vitals_copy, on='enc_id')

    # function to find which meds were currently being taken during the encounter period
    def encode_meds(self):
        self.meds['start_date'] = pd.to_datetime(self.meds['start_date'], format='%Y%m%d')
        self.meds['date_stopped'] = pd.to_datetime(self.meds['date_stopped'], format='%Y%m%d')

        self.meds['is_currently_taking'] = False

        def check_meds(row):
            patient_medical_history = self.meds[(self.meds['person_id'] == row['person_id'])
                                                & (self.meds['enc_id'] == row['enc_id'])
                                                & (self.meds['start_date'] < row[
                'EncounterDate'])
                                                & (self.meds['date_stopped'] > row[
                'EncounterDate'])].index
            if len(patient_medical_history) > 0:
                self.meds.iloc[patient_medical_history, -1] = True

        # apply function as a reverse lookup..
        # first check each appt date, then see which meds patient was taking on that appt
        # store output as boolean for that medication
        self.encounters.apply(check_meds, axis=1)


    def merge_meds(self):
        # note that the meds table may or may not have columns depending on sample
        meds_wide = pd.get_dummies(self.meds[['enc_id', 'medid', 'is_currently_taking']]
                                   .query('is_currently_taking'), columns=['medid']) \
            .groupby('enc_id', as_index=False).max()

        self.main = self.main.merge(meds_wide, on='enc_id', how='left')

        # not all patients have active meds...take care to fill those nulls
        self.main[[col for col in meds_wide.columns if col != 'enc_id']].fillna(0, inplace=True)


    # reading in raw data from all datasets
    def generate_csv_attributes(self):
        self.encounters = pd.read_csv(self.data_path + '1_BaseEncounters_Dempgraphics_Payers.csv')
        self.format_encounters()  # align encounters table columns...make sure columns are aligned as planned
        
        self.cpt = pd.read_csv(self.data_path + '2_CPT_Codes.csv')
        self.vitals = pd.read_csv(self.data_path + '3_vitals_signs.csv')
        self.meds = pd.read_csv(self.data_path + '4_patient_medication.csv')
        self.labs = pd.read_csv(self.data_path + '5_lab_nor__lab_results_obr_p__lab_results_obx.csv')
        self.diagnosis = pd.read_csv(self.data_path + '6_patient_diagnoses.csv')
        self.assessments = pd.read_csv(self.data_path + '7_assessment_impression_plan_.csv')


    # createa a wide table out of the labs table...
    # perform encodings, etc...
    def format_labs(self, encoded=False):
        # %% Remove Deleted rows and drop deleted indicators
        self.labs = self.labs[self.labs['lab_nor_delete_ind'] == 'N']
        self.labs = self.labs.drop(columns=['lab_nor_delete_ind', 'lab_results_obx_delete_ind'])

        # Remove incomplete labs & drop column
        self.labs = self.labs[self.labs['lab_nor_completed_ind'] == 'Y']
        self.labs = self.labs.drop(columns=['lab_nor_completed_ind'])
        # Remove pending labs
        self.labs = self.labs[self.labs['lab_nor_test_status'] != 'InProcessUnspecified']
        self.labs = self.labs[self.labs['lab_nor_test_status'] != 'Pending']

        # Set lab_results to result description if exists, otherwise use test description
        self.labs['lab_results'] = np.where(
            self.labs['lab_results_obx_result_desc'].notnull(),
            self.labs['lab_results_obx_result_desc'],
            self.labs['lab_results_obr_p_test_desc']
        )

        # Combine outcome with test result
        self.labs['lab_results'] = np.select(
            [  # condition
                self.labs.lab_results_obx_abnorm_flags == '>',
                self.labs.lab_results_obx_abnorm_flags == 'H',
                self.labs.lab_results_obx_abnorm_flags == 'HH',
                self.labs.lab_results_obx_abnorm_flags == '<',
                self.labs.lab_results_obx_abnorm_flags == 'L',
                self.labs.lab_results_obx_abnorm_flags == 'LL',
                self.labs.lab_results_obx_abnorm_flags == 'A'
            ],
            [  # value
                'HIGH ' + self.labs['lab_results'],
                'HIGH ' + self.labs['lab_results'],
                'VERY HIGH ' + self.labs['lab_results'],
                'LOW ' + self.labs['lab_results'],
                'LOW ' + self.labs['lab_results'],
                'VERY LOW ' + self.labs['lab_results'],
                'ABNORMAL ' + self.labs['lab_results']
            ],
            default='NORMAL'
        )

        # Capture abnormal labs
        abnormal_labs = self.labs[self.labs['lab_results'] != 'NORMAL']
        abnormal_labs['lab_results'] = abnormal_labs['lab_results'].str.title()
        abnormal_labs = pd.DataFrame(abnormal_labs[['lab_nor_person_id', 'lab_nor_enc_id', 'lab_results']].groupby(
            ['lab_nor_person_id', 'lab_nor_enc_id'])['lab_results'].apply(set))

        abnormal_labs.reset_index(inplace=True)
        abnormal_labs.columns = ['person_id', 'enc_id', 'lab_results']

        if not encoded:
            self.labs = abnormal_labs

        # Pandas get_dummies function will not parse lists, enter the multiLabelBinarizer
        mlb = MultiLabelBinarizer()
        encoded_labs = abnormal_labs.join(
            pd.DataFrame(mlb.fit_transform(abnormal_labs['lab_results']), columns=mlb.classes_))

        self.labs = encoded_labs


    def merge_labs(self, rename=True):
        labs_copy = self.labs.copy()
        labs_copy.drop(columns=['person_id', 'lab_results'], inplace=True)
        if rename:
            labs_copy =  self.rename_cols(labs_copy, prefix='lab_')
        self.main = self.main.merge(labs_copy, on='enc_id', how='left')
        
        # TODO: address null values col
        self.main[[col for col in labs_copy.columns if col != 'enc_id']].fillna(0, inplace=True)


    # function to transform assessments table to merge with encounters
    def format_assessments(self):
        assessment = self.assessments
        # Filter for relevant enc_ids
        assessment = assessment[assessment['enc_id'].isin(
            set(assessment['enc_id']).intersection(
                set(self.encounters.enc_id)))]

        # Collapse data down by enc_id for assessment

        assessment_text = assessment.groupby(['person_id','enc_id'])['txt_description'].apply(list)
        assessment_codeID = assessment.groupby(['person_id','enc_id'])['txt_diagnosis_code_id'].apply(list)

        assessment_text = pd.DataFrame(assessment_text)
        assessment_codeID = pd.DataFrame(assessment_codeID)

        # Merge series data from text and codeID columns into one df for assessment
        assessment2 = assessment_text.merge(assessment_codeID, how = 'left', on = ['person_id','enc_id'])
        assessment2 = pd.DataFrame(assessment2)
        assessment2.reset_index(inplace=True)

        # Remove punctuation, convert all to lowercase, remove stopwords, tokenize, create bigrams

        # Remove Punctuation and convert to lower
        assessment2['txt_description'] = assessment2.txt_description.apply(
            lambda x: ', '.join([str(i) for i in x]))
        assessment2['txt_description'] = assessment2['txt_description'].str.replace('[^\w\s]','')
        assessment2['txt_description'] = assessment2['txt_description'].str.lower()

        #tokenize
        assessment2['txt_tokenized'] = assessment2.apply(
            lambda row: nltk.word_tokenize(row['txt_description']), axis=1)

        #Remove Stopwords
        stop = stopwords.words('english')
        assessment2['txt_tokenized'] = assessment2['txt_tokenized'].apply(
            lambda x: [item for item in x if item not in stop])

        #Create ngrams
        assessment2['ngrams'] = assessment2.apply(
            lambda row: list(nltk.trigrams(row['txt_tokenized'])),axis=1) 

        # Convert trigram lists to words joined by underscores
        assessment2['ngram2'] = assessment2.ngrams.apply(lambda row:['_'.join(i) for i in row])

        # Convert trigram and token lists to strings
        assessment2['txt_tokenized2'] = assessment2['txt_tokenized'].apply(' '.join)
        assessment2['ngram2'] = assessment2.ngram2.apply(lambda x: ' '.join([str(i) for i in x]))

        # Get noun phrases
        nlp = en_core_web_sm.load()

        def getNounChunks(text_data):
            doc = nlp(text_data)
            noun_chunks = list(doc.noun_chunks)
            noun_chunks_strlist = [chunk.text for chunk in noun_chunks]
            noun_chunks_str = '_'.join(noun_chunks_strlist)
            return noun_chunks_str

        assessment2['np_chunks'] = assessment2['txt_tokenized2'].apply(getNounChunks)

        # Pair down assessments table to columns of interest
        #assessment2 = assessment2[['person_id','enc_id','txt_description','txt_tokenized','ngrams','ngram2','txt_tokenized2','np_chunks']]

        # DBSCAN Clusterin for trigrams and noun phrase chunks
        tfidf = TfidfVectorizer()

        #tfidf_data_ngram = tfidf.fit_transform(assessment2['ngram2'])
        tfidf_data_np = tfidf.fit_transform(assessment2['np_chunks'])

        cluster_model = KMeans(n_jobs=-1,n_clusters=15)

        #print("Starting ngram KMeans model fit...")
        #ngram_db_model = cluster_model.fit(tfidf_data_ngram)
        #print("ngram KMeans model fit COMPLETE...")

        print("Starting NP Chunk Kmeans model fit on np chunks...")
        np_db_model = cluster_model.fit(tfidf_data_np)
        print("NP Chunk Kmeans model fit on np chunks COMPLETE...")


        # KMeans cluster counts and labeling
        # assessment2['ngram_clusters'] = ngram_db_model.labels_
        assessment2['np_chunk_clusters'] = np_db_model.labels_

        print("ngram Model Cluster Count:",assessment2['ngram_clusters'].nunique())
        print("ngram DBSCAN Model Cluster Count:",assessment2['np_chunk_clusters'].nunique())

        #%% LDA clustering
        from sklearn.decomposition import LatentDirichletAllocation as LDA
        from sklearn.feature_extraction.text import CountVectorizer

        count_vectorizer =CountVectorizer()
        count_data = count_vectorizer.fit_transform(assessment2['ngram2'].values.astype('U'))
        lda = LDA(n_components = 20,learning_method = 'online')
        lda.fit(count_data)


        # LDA Cluster labeling
        topic_values = lda.transform(count_data)
        assessment2['topic_clusters'] = topic_values.argmax(axis=1)


        #%% FINAL ASSESSMENTS TABLE
        #assessment2.drop(['np_chunk_clusters','topic_clusters','txt_description','txt_tokenized','ngrams','ngram2','txt_tokenized2','np_chunks'],axis=1, inplace=True)

        kmeans_cluster = pd.get_dummies(assessment2.np_chunk_clusters, prefix='kmeans')
        topic_cluster = pd.get_dummies(assessment2.topic_clusters, prefix='topic')

        # use pd.concat to join the new columns with your original dataframe
        assessment2 = pd.concat([assessment2,kmeans_cluster],axis=1)
        assessment2 = pd.concat([assessment2,topic_cluster],axis=1)


        # Read in diagnosis table
        diagnoses = pd.read_csv(self.data_path + '6_patient_diagnoses.csv')

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

        # Merge series data from text and codeID columns into one df for assessment
        diagnoses2 = diagnosis_icd9 \
            .merge(diagnosis_dc, how = 'left', on = ['person_id','enc_id']) \
            .merge(diagnosis_desc, how = 'left', on = ['person_id','enc_id']) \
            .merge(diagnosis_datesymp, how = 'left', on = ['person_id','enc_id']) \
            .merge(diagnosis_datediag, how = 'left', on = ['person_id','enc_id']) \
            .merge(diagnosis_dateresl, how = 'left', on = ['person_id','enc_id']) \
            .merge(diagnosis_statusid, how = 'left', on = ['person_id','enc_id']) \
            .merge(diagnosis_dx, how = 'left', on = ['person_id','enc_id']) \
            .merge(diagnosis_chronic, how = 'left', on = ['person_id','enc_id']) \
            .merge(diagnosis_rcdelswhr, how = 'left', on = ['person_id','enc_id'])

        diagnoses2 = pd.DataFrame(diagnoses2)
        diagnoses2.reset_index(inplace=True)

        # Merge assements[txt_description] to ngd df
        #Diagnosis occur after assessments, diagnosis table is smaller than assessments table
        assessments_diagnoses = assessment2.merge(diagnoses2, how = 'left', on = ['person_id','enc_id'])

        self.asmt_diag = assessments_diagnoses
        # drop 5 rows with missing diag data
        self.asmt_diag = self.asmt_diag[~self.asmt_diag['diagnosis_code_id'].isnull()]


    def one_hot(self, df, col_name, prefix=''):
        mlb = MultiLabelBinarizer()
        df = df.join(pd.DataFrame(mlb.fit_transform(df[col_name]),columns=prefix+mlb.classes_))
        df = df.drop(columns=[col_name])
        return df

    def merge_assessments(self, rename=True):
        assess_copy = self.asmt_diag.copy().fillna(0)

        # Pair down assessments table to columns of interest
        assess_copy = assess_copy[['person_id','enc_id','txt_description','txt_tokenized','ngrams','ngram2','txt_tokenized2','np_chunks','diagnosis_code_id','description']]
        assess_copy = self.one_hot(assess_copy, 'diagnosis_code_id', prefix='icd_')

        if rename:
            assess_copy = self.rename_cols(assess_copy, prefix='asmt_')
        self.main = self.main.merge(assess_copy, on=['person_id','enc_id'], how='left')


    def encode_encounters(self):
        # Apply enc_ label to encounter columns
        self.main.rename(columns = { 
            'place_of_service':'enc_place_of_service',
            'Provider_id':'enc_Provider_id',
            'EncounterDate':'enc_EncounterDate',
            'Race':'enc_Race',
            'Ethnicity':'enc_Ethnicity',
            'Gender':'enc_Gender',
            'AgeAtEnc':'enc_AgeAtEnc',
            'VisitType':'enc_VisitType',
            'ServiceDepartment':'enc_ServiceDepartment',
            'LocationName':'enc_LocationName',
            'Reason_for_Visit':'enc_Reason_for_Visit',
            'CPT_Code':'enc_CPT_Code',
            'CPT_Code_Seq':'enc_CPT_Code_Seq',
            'Encounter_Primary_payer':'enc_Primary_payer',
            'Encounter_Secondary_Payer':'enc_Secondary_Payer',
            'Encounter_Teritiary_Payer':'enc_Teritiary_Payer'
        }, inplace=True)

        # Update EncounterDate to be an ordinal date (see: pandas.Timestamp.toordinal)
        self.main['enc_EncounterDate'] = self.main['enc_EncounterDate'].apply(lambda x: x.toordinal())

        self.main = pd.get_dummies(self.main, columns = [
            'enc_Provider_id',
            'enc_Race',
            'enc_Ethnicity',
            'enc_Gender',
            'enc_VisitType',
            'enc_ServiceDepartment',
            'enc_LocationName',
            'enc_Reason_for_Visit',
            'enc_Primary_payer',
            'enc_Secondary_Payer',
            'enc_Teritiary_Payer'
        ])


    def clean(self):
        # Drop single value columns
        single_val_columns = []
        for col in self.main:
            try:
                if self.main[col].nunique() == 1:
                    single_val_columns.append(col)
            except TypeError:
                pass # skip list cols
        self.main.drop(columns=single_val_columns, inplace=True)

        # CPT is already encoded via the CPT table
        self.main.drop(columns='enc_CPT_Code', inplace=True)

        # TODO moar clean plz


    # return the main data output
    def create(self, name='main'):
        # generating data attributes
        self.generate_csv_attributes()

        # step 1...make sure alzheimers and dementia response is encoded
        self.encode_alzheimers()

        # step 2...add on cpt table
        self.merge_cpt()

        # step 3...load vitals table onto main
        self.merge_vitals()

        # step 4...encode medications to find current meds...join onto the main for medication list
        self.encode_meds()
        self.merge_meds()

        # step 5...load labs onto the main dataframe
        self.format_labs(encoded=True)
        self.merge_labs()

        # step 6...load merged assessments + diagnoses onto main dataframe
        self.format_assessments()
        self.merge_assessments()

        # step 7...clean data: drop NAs, rename cols etc
        self.encode_encounters()
        self.clean()

        # write to pickle file
        self.write(filename=name)


    # helper function write entire class object
    def write(self, filename='main'):
        with open(self.data_path + filename + '.pickle', 'wb') as picklefile:
            pickle.dump(self, picklefile)


    # helper function to return entire class object
    def load(self, filename='main'):
        with open(self.data_path + filename + '.pickle', 'rb') as picklefile:
            return pickle.load(picklefile)


    # helper to add prefix to colnames:
    def rename_cols(self, df, prefix=''):
        new_cols = []
        for c in list(df):
            if c in ['person_id','enc_id']:
                new_cols.append(c)
            else:
                new_cols.append(prefix+c)
        df.columns = new_cols
        return df


if __name__ == "__main__":
    data = DataLoader(subset=1000)
    data.create()
    print(data.load())



# %%
