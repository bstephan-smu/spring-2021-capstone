# %% Requirements:
import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords


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

        # store list of dementia icd codes here... TODO will be updated in later function
        self.dementia_icd_codes = None

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

    # Fix header on encounters:
    def format_encounters(self):
        self.align_columns()  # ensure that encounters columns are properly aligned
        self.encounters['EncounterDate'] = pd.to_datetime(self.encounters['EncounterDate'], format='%Y%m%d')

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

    # function to find which meds were currently being taken during the encounter period
    def encode_meds(self):
        self.meds['start_date'] = pd.to_datetime(self.meds['start_date'], format='%Y%m%d')
        self.meds['date_stopped'] = pd.to_datetime(self.meds['date_stopped'], format='%Y%m%d')

        self.meds['is_currently_taking'] = False
        x = 1

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
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        encoded_labs = abnormal_labs.join(
            pd.DataFrame(mlb.fit_transform(abnormal_labs['lab_results']), columns=mlb.classes_))

        self.labs = encoded_labs

    # function to transform assessments table to merge with encounters
    def format_assessment(self):
        assessment_text = pd.DataFrame(self.assessments.groupby(['person_id', 'enc_id'])['txt_description'].apply(list))
        assessment_codeID = pd.DataFrame(self.assessments.groupby(['person_id', 'enc_id'])['txt_diagnosis_code_id'].apply(list))

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

    # return the main data output
    def create(self):
        # generating data attributes
        self.generate_csv_attributes()

        # step 1...make sure alzheimers and dementia response is encoded
        self.encode_alzheimers()

        # step 2...add on cpt table
        df_cpt_codes_encoded = pd.concat(
            [self.cpt[['enc_id']], pd.get_dummies(self.cpt['CPT_Code'], drop_first=True, prefix='cpt')],
            axis=1) \
            .groupby('enc_id', as_index=False).max()

        main = self.encounters.merge(df_cpt_codes_encoded, on='enc_id')

        # step 3...load vitals table onto main
        # get average vital measurement per patient encounter
        vitals_agg = self.vitals[['enc_id', 'BMI_calc', 'bp_diastolic', 'bp_systolic',
                                  'height_cm', 'pulse_rate', 'respiration_rate', 'temp_deg_F',
                                  'weight_lb']].groupby('enc_id', as_index=False).max()

        main = main.merge(vitals_agg, on='enc_id')

        # step 4...encode medications to find current meds...join onto the main for medication list
        self.encode_meds()
        # note that the meds table may or may not have columns depending on sample
        meds_wide = pd.get_dummies(self.meds[['enc_id', 'medid', 'is_currently_taking']]
                                   .query('is_currently_taking'), columns=['medid']) \
            .groupby('enc_id', as_index=False).max()

        main = main.merge(meds_wide, on='enc_id', how='left')

        # not all patients have active meds...take care to fill those nulls
        main[[col for col in meds_wide.columns if col != 'enc_id']].fillna(0, inplace=True)

        # step 5...load labs onto the main dataframe
        self.format_labs(encoded=True)
        labs_copy = self.labs.copy()
        labs_copy.drop(columns=['person_id', 'lab_results'], inplace=True)
        main = main.merge(labs_copy, on='enc_id', how='left')
        # TODO: address null values col
        main[[col for col in labs_copy.columns if col != 'enc_id']].fillna(0, inplace=True)

        # step 6...load assessments onto main dataframe
        assessments = self.format_assessment()

        # step 7...load diagnosis onto main dataframe

        # double check to see if jeff has converted this portion yet.

        # write to pickle file
        self.write(main)

    # helper function write main dataframe
    def write(self, df, name='main'):
        df.to_pickle(self.data_path + 'main')

    # helper function to return main dataframe
    def load(self, name='main'):
        return pd.read_pickle(self.data_path + 'main')


if __name__ == "__main__":
    data = DataLoader(subset=1000)
    data.create()
    print(data.load())
