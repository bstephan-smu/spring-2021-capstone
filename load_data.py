# %% Requirements:
import pandas as pd
import numpy as np
import re
import os
import pickle
import nltk
from nltk.corpus import stopwords
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
        self.main = None

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
        self.main = self.encounters.copy()


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
        #TODO: replace this code block with Jeff's new code

        assessment2 = pd.read_csv(self.data_path + 'assessments_diagnoses_join2.csv')
        # %% Pair down assessments table to columns of interest
        assessment2 = assessment2[[
            'person_id', 
            'enc_id', 
            'txt_description', 
            'txt_tokenized', 
            'ngrams', 
            'ngram2', 
            'txt_tokenized2'
            ]]

        return assessment2


    def merge_assessments(self, rename=True):
        assess_copy = self.assessments.copy()
        if rename:
            assess_copy = self.rename_cols(assess_copy, prefix='as_')
        self.main = self.main.merge(assess_copy, on='enc_id', how='left')


    # return the main data output
    def create(self):
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

        # write to pickle file
        self.write()


    # helper function write entire class object
    def write(self, name='main'):
        with open(self.data_path + name + '.pickle', 'wb') as picklefile:
            pickle.dump(self, picklefile)


    # helper function to return entire class object
    def load(self, name='main'):
        with open(self.data_path + name + '.pickle', 'rb') as picklefile:
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

