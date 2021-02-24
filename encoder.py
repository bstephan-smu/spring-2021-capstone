from scipy.sparse import data
from load_data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nlp_utils

class Encoder(DataLoader):
    def __init__(self, filename='main'):
        self.create(filename)
        self.main = self.encounters.copy()


    def one_hot(df:pd.DataFrame, col_name:str, prefix:str=''):
        # Pandas get_dummies function will not parse lists, enter the multiLabelBinarizer
        mlb = MultiLabelBinarizer()
        df = df.join(pd.DataFrame(mlb.fit_transform(df[col_name]), columns=prefix + mlb.classes_))
        df = df.drop(columns=[col_name])
        return df


    # helper to add prefix to colnames:
    def rename_cols(df, prefix=''):
        new_cols = []
        for c in list(df):
            if c in ['person_id', 'enc_id']:
                new_cols.append(c)
            else:
                new_cols.append(prefix + c)
        df.columns = new_cols
        return df


    def get_encounters(self):
        # CPT is already encoded via the CPT table, and drop cols unrelated to response:
        dropcols = ['Encounter_Primary_Payer', 'Encounter_Secondary_Payer', 'Encounter_Teritiary_Payer',
                    'LocationName', 'ServiceDepartment', 'VisitType', 'CPT_Code', 'CPT_Code_Seq', 'Provider_id',
                    'place_of_service']

        self.main.drop(columns=dropcols, inplace=True)
        
        # drop the row that have NA's from ANY of the following columns 'Race', 'Gender', 'AgeAtEnc'
        # Its important to have age race and gender for each entry        
        self.main.dropna(subset=['Race', 'Gender', 'AgeAtEnc'])

        # Apply enc_ label to encounter columns
        self.main.rename(columns={
            'EncounterDate': 'enc_EncounterDate',
            'Race': 'enc_Race',
            'Ethnicity': 'enc_Ethnicity',
            'Gender': 'enc_Gender',
            'AgeAtEnc': 'enc_AgeAtEnc',
            'rfv_cluster':'enc_rfv_cluster'
        }, inplace=True)

        # Update EncounterDate to be an ordinal date (see: pandas.Timestamp.toordinal)
        self.main['enc_EncounterDate'] = self.main['enc_EncounterDate'].apply(lambda x: x.toordinal())

        # Onehot encode 
        self.main = pd.get_dummies(self.main, columns=[
            'enc_Race',
            'enc_Ethnicity',
            'enc_Gender',
            'enc_rfv_cluster'
        ])


    def get_labs(self, rename=True):
        self.labs = self.one_hot('labs', 'lab_results', 'lab_')
        labs_copy = self.labs.copy()
        labs_copy.drop(columns=['person_id', 'lab_results'], inplace=True)
        if rename:
            labs_copy = self.rename_cols(labs_copy, prefix='lab_flag_')
        self.main = self.main.merge(labs_copy, on='enc_id', how='left')

        # TODO: address null values col
        self.main[[col for col in labs_copy.columns if col != 'enc_id']].fillna(0, inplace=True)


    def get_labs_continuous(self):
        labs2_encoded = self.labs_cont.groupby(['enc_id', 'lab_test'])['lab_test_results'].aggregate(
            'mean').unstack().reset_index()
        self.main = self.main.merge(labs2_encoded, how='left', on='enc_id')


    def get_meds(self):
        # note that the meds table may or may not have columns depending on sample
        meds_wide = pd.get_dummies(self.meds[['enc_id', 'med', 'is_currently_taking']]
                                   .query('is_currently_taking'), columns=['med']) \
            .groupby('enc_id', as_index=False).max()

        self.main = self.main.merge(meds_wide, on='enc_id', how='left')

        # not all patients have active meds...take care to fill those nulls
        self.main[[col for col in meds_wide.columns if col != 'enc_id']].fillna(0, inplace=True)


    def get_cpt(self):
        df_cpt_codes_encoded = pd.concat(
            [
                self.cpt[['enc_id']],
                pd.get_dummies(self.cpt['CPT_Code'], drop_first=True, prefix='cpt')
            ], axis=1) \
            .groupby('enc_id', as_index=False).max()

        self.main = self.main.merge(df_cpt_codes_encoded, on='enc_id')


    def get_reason_for_visit(self):
        x = self.encounters['Reason_for_Visit']  # grabbing reason for visit and subsetting
        x = x.str.lower().tolist()  # setting all text to lowercase
        splits = nlp_utils.split_numbers(x)
        encoded = nlp_utils.custom_lexicon(splits)
        only_alpha = nlp_utils.strip_non_alpha(encoded)
        stemmed = nlp_utils.porter_stemmer(only_alpha)
        tfidf_ = TfidfVectorizer(stop_words={'english'})
        tfidf = tfidf_.fit_transform(stemmed)

        # ask user for optimal values
        km = KMeans(n_clusters=22, init='k-means++', max_iter=100, n_init=1)
        km.fit(tfidf)
        self.main['rfv_cluster'] = km.labels_


    def get_vitals(self, rename=True):
        # get average vital measurement per patient encounter
        vitals_agg = self.vitals[['enc_id', 'BMI_calc', 'bp_diastolic', 'bp_systolic',
                                  'height_cm', 'pulse_rate', 'respiration_rate', 'temp_deg_F',
                                  'weight_lb']].groupby('enc_id', as_index=False).max()

        vitals_copy = vitals_agg.copy()
        vitals_copy = self.rename_cols(vitals_copy, prefix='vit_')
        self.main = self.main.merge(vitals_copy, on='enc_id')


    def get_assessments(self, rename=True):
        #TODO : Do we REALLY want to fill NA here?
        assess_copy = self.asmt_diag.copy().fillna(0)

        # Pair down assessments table to columns of interest
        assess_copy = assess_copy[
            ['person_id', 'enc_id', 'txt_description', 'txt_tokenized', 'ngrams', 'ngram2', 'txt_tokenized2',
             'np_chunks', 'CCSR Category', 'description']]

        def stripNA(lst_col):
            for item in lst_col:
                if type(item) != str:
                    lst_col.remove(item)
            return lst_col

        assess_copy['CCSR Category2'] = assess_copy['CCSR Category'].apply(stripNA)
        assess_copy.drop(columns=['CCSR Category'], inplace = True)

        assess_copy = one_hot(assess_copy, 'CCSR Category2', prefix='ccsr_')

        if rename:
            assess_copy = self.rename_cols(assess_copy, prefix='asmt_')
        self.main = self.main.merge(assess_copy, on=['person_id', 'enc_id'], how='left')


    def get_clusters(self):
        # Add in clusters:
        assess_copy = self.asmt_diag.copy()
        assess_cluster_cols = [col for col in assess_copy if col.startswith('topic') or col.startswith('kmeans')]
        assess_cluster_cols += ['person_id','enc_id','np_chunk_clusters']
        clusters = assess_copy[assess_cluster_cols]
        clusters = self.rename_cols(assess_copy, prefix='asmt_')
        self.main = self.main.merge(clusters, on=['person_id', 'enc_id'], how='left')


def main():
    pass


if __name__ == "__main__":
    main()