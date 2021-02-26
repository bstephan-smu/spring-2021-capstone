from scipy.sparse import data
from load_data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nlp_utils
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
class Encoder(DataLoader):
    def __init__(self):
        super().__init__() 


    def one_hot(self, df:pd.DataFrame, col_name:str, prefix:str=''):
        # Pandas get_dummies function will not parse lists, enter the multiLabelBinarizer
        mlb = MultiLabelBinarizer()
        df = df.join(pd.DataFrame(mlb.fit_transform(df[col_name]), columns=prefix + mlb.classes_))
        df = df.drop(columns=[col_name])
        return df


    # helper to add prefix to colnames:
    def rename_cols(self, df, prefix=''):
        new_cols = []
        for c in list(df):
            if c in ['person_id', 'enc_id']:
                new_cols.append(c)
            else:
                new_cols.append(prefix + c)
        df.columns = new_cols
        return df


    #TODO: Need to modify this to account for when all items are removed from the list and it returns an empty list
    def stripNA(self, lst_col):  
        for item in lst_col:
            if type(item) != str:
                lst_col.remove(item)
        return lst_col


    def get_encounters(self):
        # CPT is already encoded via the CPT table, and drop cols unrelated to response:
        dropcols = ['Encounter_Primary_Payer', 'Encounter_Secondary_Payer', 'Encounter_Teritiary_Payer',
                    'LocationName', 'ServiceDepartment', 'VisitType', 'CPT_Code', 'CPT_Code_Seq', 'Provider_id',
                    'place_of_service']

        self.main.drop(columns=dropcols, inplace=True)
        
        # drop the row that have NA's from ANY of the following columns 'Race', 'Gender', 'AgeAtEnc'
        # Its important to have age race and gender for each entry        
        self.main.dropna(subset=['race_ethincity', 'Gender', 'AgeAtEnc'])

        # Apply enc_ label to encounter columns
        self.main.rename(columns={
            'EncounterDate': 'enc_EncounterDate',
            'Gender': 'enc_Gender',
            'AgeAtEnc': 'enc_AgeAtEnc',
            'race_ethincity': 'enc_RaceEth'
        }, inplace=True)

        # Update EncounterDate to be an ordinal date (see: pandas.Timestamp.toordinal)
        self.main['enc_EncounterDate'] = self.main['enc_EncounterDate'].apply(lambda x: x.toordinal())

        # Onehot encode 
        self.main = pd.get_dummies(self.main, columns=['enc_Gender'], drop_first=True)
        self.main = pd.get_dummies(self.main, columns=['enc_RaceEth'], drop_first=False)


    def get_labs(self):
        self.labs = self.one_hot(self.labs, 'lab_results', '')
        labs_copy = self.labs.copy()
        labs_copy.drop(columns=['person_id'], inplace=True)
        labs_copy = self.rename_cols(labs_copy, prefix='lab_')
        self.main = self.main.merge(labs_copy, on='enc_id', how='left')

        # TODO: address null values col
        self.main[[col for col in labs_copy.columns if col != 'enc_id']].fillna(0, inplace=True)
        # Fix float cols
        cols = [col for col in self.main if col.startswith('lab_')]
        self.main[cols] = self.main[cols].fillna(value=0).astype(int)

    def get_labs_continuous(self):
        labs2_encoded = self.labs_cont.groupby(['enc_id', 'lab_test'])['lab_test_results'].aggregate(
            'mean').unstack().reset_index()
        self.main = self.main.merge(labs2_encoded, how='left', on='enc_id')
        # Fix float cols
        cols = [col for col in self.main if col.startswith('asmt_')]
        self.main[cols] = self.main[cols].fillna(value=0).astype(int)

    def get_meds(self):
        # note that the meds table may or may not have columns depending on sample
        meds_wide = pd.get_dummies(self.meds[['enc_id', 'med', 'is_currently_taking']]
                                   .query('is_currently_taking'), columns=['med']) \
            .groupby('enc_id', as_index=False).max()

        self.main = self.main.merge(meds_wide, on='enc_id', how='left')

        # not all patients have active meds...take care to fill those nulls
        medcols = [col for col in self.main if col.startswith('med_')]
        self.main[medcols] = self.main[medcols].fillna(value=0).astype(int)


    def get_cpt(self):
        cpt_desc = pd.read_csv('E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/cpt_updated_full.csv')
        cpt_desc.rename(columns={'cpt':'CPT_Code', 'short_description':'cpt_desc'}, inplace=True)
        self.cpt = self.cpt.merge(cpt_desc, how='left', on = 'CPT_Code')
        df_cpt_codes_encoded = pd.concat(
            [
                self.cpt[['enc_id']],
                pd.get_dummies(self.cpt['cpt_desc'], prefix='cpt')
            ], axis=1) \
            .groupby('enc_id', as_index=False).max()

        self.main = self.main.merge(df_cpt_codes_encoded, on='enc_id')


    def get_reason_for_visit(self):
        x = self.encounters['Reason_for_Visit']  # grabbing reason for visit and subsetting
        x = x.str.lower().tolist()  # setting all text to lowercase
        splits = nlp_utils.split_numbers(self, x)
        encoded = nlp_utils.custom_lexicon(self, splits)
        only_alpha = nlp_utils.strip_non_alpha(self, encoded)
        stemmed = nlp_utils.porter_stemmer(self, only_alpha)
        tfidf_ = TfidfVectorizer(stop_words={'english'})
        tfidf = tfidf_.fit_transform(stemmed)

        # ask user for optimal values
        km = KMeans(n_clusters=22, init='k-means++', max_iter=100, n_init=1)
        km.fit(tfidf)
        self.main['rfv_cluster'] = km.labels_
        self.main = pd.get_dummies(self.main, columns=['rfv_cluster'])


    def get_vitals(self, rename=True):
        # get average vital measurement per patient encounter
        vitals_agg = self.vitals[['enc_id', 'BMI_calc', 'bp_diastolic', 'bp_systolic',
                                  'height_cm', 'pulse_rate', 'respiration_rate', 'temp_deg_F',
                                  'weight_lb']].groupby('enc_id', as_index=False).max()

        vitals_copy = vitals_agg.copy()
        vitals_copy = self.rename_cols(vitals_copy, prefix='vit_')
        self.main = self.main.merge(vitals_copy, on='enc_id')


    def get_response_cols(self, return_val='description'):
        dementia_string = '|'.join(self.dementia_lookup)
        diag = pd.read_csv(self.data_path + '6_patient_diagnoses.csv')

        dementia_output = list(
            diag[diag.description.str.contains(dementia_string, regex=True, flags=re.IGNORECASE)]
            [return_val].unique()
        )
        dementia_output = [desc for desc in dementia_output if desc not in self.exclude_dementia_lookup]
        self.dementia_icd_codes = diag[
            diag.description.isin(dementia_output)].icd9cm_code_id.unique()

        # Collect response
        AD_people = diag[
            diag.description.str.contains(self.alz_regex, regex=True, flags=re.IGNORECASE)].person_id.unique()
        AD_encounters = diag[
            diag.description.str.contains(self.alz_regex, regex=True, flags=re.IGNORECASE)].enc_id.unique()
        dem_people = diag[diag.description.isin(dementia_output)].person_id.unique()
        dem_encounters = diag[diag.description.isin(dementia_output)].enc_id.unique()

        # Set response
        self.main['AD_encounter'] = self.main.enc_id.isin(AD_encounters).astype(int)
        self.main['AD_person'] = self.main.person_id.isin(AD_people).astype(int)
        self.main['dem_encounter'] = self.main.enc_id.isin(dem_encounters).astype(int)
        self.main['dem_person'] = self.main.person_id.isin(dem_people).astype(int)
        self.main['Cognition'] = np.select(
            [self.main.AD_person == 1, self.main.dem_person == 1],
            ['AD', 'Dementia'],
            default='Normal'
        )

    def get_diagnoses(self):  # TODO: This is a hack, need to modify the stripNA function
        self.diagnosis['CCSR Category Description'].apply(self.stripNA).apply(self.stripNA).apply(self.stripNA)

        diag = self.diagnosis.copy()
        diag = self.rename_cols(diag, prefix='asmt_')
        diag = self.one_hot(diag, 'asmt_CCSR Category Description', prefix='ccsr_')

        self.main = self.main.merge(diag, on=['person_id', 'enc_id'], how='left')
        # Fix float cols
        cols = [col for col in self.main if col.startswith('ccsr_')]
        self.main[cols] = self.main[cols].fillna(value=0).astype(int)

    def build_assessment_clusters(self):
        try:
            with open(self.data_path + 'bin/assessment_with_clusters.pickle', 'rb') as picklefile:
                self.assessments = pickle.load(picklefile)

        except FileNotFoundError:
            # DBSCAN Clusterin for trigrams and noun phrase chunks
            tfidf = TfidfVectorizer()
            assessment2 = self.assessments.copy()
            # tfidf_data_ngram = tfidf.fit_transform(assessment2['ngram2'])
            tfidf_data_np = tfidf.fit_transform(assessment2['np_chunks'])

            print("Starting NP Chunk Kmeans model fit on np chunks...")
            cluster_model = KMeans(n_jobs=-1, n_clusters=15)
            np_db_model = cluster_model.fit(tfidf_data_np)
            print("NP Chunk Kmeans model fit on np chunks COMPLETE...")

            # KMeans cluster counts and labeling
            assessment2['np_chunk_clusters'] = np_db_model.labels_

            # print("ngram Model Cluster Count:",assessment2['ngram_clusters'].nunique())
            print("ngram DBSCAN Model Cluster Count:", assessment2['np_chunk_clusters'].nunique())

            # %% LDA clustering
            count_vectorizer = CountVectorizer()
            count_data = count_vectorizer.fit_transform(assessment2['ngram2'].values.astype('U'))
            lda = LDA(n_components=20, learning_method='online')
            lda.fit(count_data)

            # LDA Cluster labeling
            topic_values = lda.transform(count_data)
            assessment2['topic_clusters'] = topic_values.argmax(axis=1)

            # %% FINAL ASSESSMENTS TABLE
            # assessment2.drop(['np_chunk_clusters','topic_clusters','txt_description','txt_tokenized','ngrams','ngram2','txt_tokenized2','np_chunks'],axis=1, inplace=True)

            kmeans_cluster = pd.get_dummies(assessment2.np_chunk_clusters, prefix='kmeans')
            topic_cluster = pd.get_dummies(assessment2.topic_clusters, prefix='topic')

            # use pd.concat to join the new columns with your original dataframe
            assessment2 = pd.concat([assessment2, kmeans_cluster], axis=1)
            assessment2 = pd.concat([assessment2, topic_cluster], axis=1)
            self.assessments = assessment2
            with open(self.data_path + 'bin/assessment_with_clusters.pickle', 'wb') as picklefile:
                pickle.dump(self.assessments, picklefile)


    def get_assessment_clusters(self):
        self.build_assessment_clusters()
        # Add in clusters:
        assess_cluster_cols = [col for col in self.assessments if col.startswith('asmt_topic') or col.startswith('asmt_kmeans')]
        assess_cluster_cols += ['person_id','enc_id']
        assess_cluster_cols.remove('asmt_topic_clusters')
        clusters = self.assessments[assess_cluster_cols]
        self.main = self.main.merge(clusters, on=['person_id', 'enc_id'], how='left')
        # Fix float cols
        cols = [col for col in self.main if col.startswith('asmt_')]
        self.main[cols] = self.main[cols].fillna(value=0).astype(int)


    def clean(self):
        # Drop single value columns
        single_val_columns = []
        for col in self.main:
            try:
                if self.main[col].nunique() == 1:
                    single_val_columns.append(col)
            except TypeError:
                pass  # skip list cols
        self.main.drop(columns=single_val_columns, inplace=True)

        # Fix float column
        #self.main['is_currently_taking'] = self.main['is_currently_taking'].astype(int)

        # Handle NAs in:
        #TODO vitals


    def build(self):
        self.create()
        print('Encoding Main DataFrame\n..encoding response')
        self.get_response_cols()
        print('..encoding encounters')
        self.get_encounters()
        print('..encoding reasons for visit')
        self.get_reason_for_visit()
        print('..encoding labs')
        self.get_labs()
    #    self.get_labs_continuous()
        print('..encoding meds')
        self.get_meds()
        print('..encoding cpt')
        self.get_cpt()
        print('..encoding vitals')
        self.get_vitals()
        print('..encoding clusters')
        self.get_assessment_clusters()
        print('..encoding diagnoses')
        self.get_diagnoses()
        self.clean()
        self.write('encoder')
        print('Encoding Complete!')
        return self


    def load(self, filename='encoder'):
        try:
            with open('bin/' + filename + '.pickle', 'rb') as picklefile:
                return pickle.load(picklefile)
        except FileNotFoundError:
            print('Pickle not found. Rebuilding.')
            self.build()
        

def main():
    from encoder import Encoder
    capData = Encoder()
    capData.build()



if __name__ == "__main__":
    main()