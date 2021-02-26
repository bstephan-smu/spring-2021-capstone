# %% Requirements:
import pandas as pd
import numpy as np
import re
import os
import pickle
import nltk
import nlp_utils
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


    # read in raw data from all datasets
    def generate_csv_attributes(self):
        self.encounters = pd.read_csv(self.data_path + '1_BaseEncounters_Dempgraphics_Payers.csv')
        self.format_encounters()  # align encounters table columns...make sure columns are aligned as planned

        self.cpt = pd.read_csv(self.data_path + '2_CPT_Codes.csv')
        self.vitals = pd.read_csv(self.data_path + '3_vitals_signs.csv')
        self.meds = pd.read_csv(self.data_path + '4_patient_medication.csv')
        self.labs = pd.read_csv(self.data_path + '5_lab_nor__lab_results_obr_p__lab_results_obx.csv')
        self.diagnosis = pd.read_csv(self.data_path + '6_patient_diagnoses.csv')
        self.assessments = pd.read_csv(self.data_path + '7_assessment_impression_plan_.csv')


    def format_encounters(self):
        # Align Columns / Fix Header
        self.encounters.drop(columns=['Encounter_Teritiary_Payer'], inplace=True)
        self.encounters.columns = ['person_id', 'enc_id', 'place_of_service', 'Provider_id', 'EncounterDate', 'Race',
                                'Ethnicity', 'Gender', 'AgeAtEnc', 'VisitType', 'ServiceDepartment',
                                'LocationName',
                                'Reason_for_Visit', 'CPT_Code', 'CPT_Code_Seq', 'Encounter_Primary_Payer',
                                'Encounter_Secondary_Payer', 'Encounter_Teritiary_Payer']

        # Set EncounterDate to datetime
        self.encounters['EncounterDate'] = pd.to_datetime(self.encounters['EncounterDate'], format='%Y%m%d')

        # in event that you'd like to subset less than total appt in data
        if self.subset is not None:
            self.encounters = self.encounters.head(self.subset)  # making a smaller sample of data for speed in dev environment

        # make sure that blank inputs are labeled as "none provided"...setting default
        self.encounters.loc[self.encounters['Reason_for_Visit'].isnull(), "Reason_for_Visit"] = "None Provided"
        
        def clean_race_ethnicity():
            """
            function to rename and clean up the race and ethnicity by combining these two
            into a single race_ethnicity column and then rename the different levels to 
            common name reducing the number of levels
            """
            self.encounters['r&e'] = self.encounters['Race'] + self.encounters['Ethnicity']
            self.encounters['race_ethincity']=self.encounters['r&e'].replace([
                #concatinated column values 
                'White Hispanic or Latino ',
                'White Declined To Specify ', 
                'White Not Hispanic or Latino ',
                'Black or African American Not Hispanic or Latino ',
                'Declined To Specify Not Hispanic or Latino ',
                'White Unknown / Not Reported ',
                'Declined to Specify Not Hispanic or Latino ',
                'Asian Declined To Specify ',
                'Declined to Specify Declined To Specify ',
                'Black or African American Declined To Specify ',
                'American Indian or Alaska Native Not Hispanic or Latino ',
                'Asian Not Hispanic or Latino ',
                'White Declined to Specify ', 
                'Native Hawaiian or Other Pacific Islander Not Hispanic or Latino ',
                'Declined To Specify Declined To Specify ',
                'Black or African American Unknown / Not Reported ',
                'Black or African American Hispanic or Latino ',
                'Declined To Specify Hispanic or Latino ',
                'Declined to Specify Unknown / Not Reported ',
                'Black or African American Declined to Specify ',
                'Declined to Specify Hispanic or Latino ',
                'Declined To Specify Declined to Specify ',
                'American Indian or Alaska Native Declined To Specify ',
                'Declined To Specify Unknown / Not Reported ',
                'Asian Unknown / Not Reported ',
                'American Indian or Alaska Native Hispanic or Latino ',
                'Asian Hispanic or Latino ',
                ' Declined To Specify '
                ],
                #these are the renames for each of the concatinated columns above
                ['White, Hispanic or Latino',				
                'White, Not Hispanic or Latino',				
                'White, Not Hispanic or Latino',				
                'Black or African American',				
                'Declined To Specify',				
                'White Not Hispanic or Latino',				
                'Declined To Specify',				
                'Asian',			
                'Declined To Specify',				
                'Black or African American',				
                'American Indian or Alaska Native',				
                'Asian',				
                'White, Not Hispanic or Latino',				
                'Native Hawaiian or Other Pacific Islander',				
                'Declined To Specify',				
                'Black or African American',				
                'Black or African American, Hispanic or Latino',				
                'Hispanic or Latino',				
                'Declined To Specify',				
                'Black or African American',				
                'Hispanic or Latino',				
                'Declined To Specify',				
                'American Indian or Alaska Native',				
                'Declined To Specify',				
                'Asian',			
                'American Indian or Alaska Native, Hispanic or Latino',				
                'Asian, Hispanic or Latino',				
                'Declined To Specify'				
                ]) 
            #drop the extra and old columns. 
            self.encounters.drop(columns=['r&e', 'Race', 'Ethnicity'], axis=1, inplace=True)
        clean_race_ethnicity()


    # function to find which meds were currently being taken during the encounter period
    def format_meds(self):
        try:
            with open(self.data_path + 'bin\meds.pickle', 'rb') as picklefile:
                self.meds = pickle.load(picklefile)

        except FileNotFoundError:
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
            with open(self.data_path + 'bin\meds.pickle', 'wb') as picklefile:
                pickle.dump(self.meds, picklefile)

        def get_med_name(med):
            med = re.search('(\\D+)(.*)', med, flags=re.IGNORECASE).group(1).strip().lower()
            if med.endswith('-'):
                med = med[:-1]
            return med

        self.meds['med'] = self.meds['medication_name'].apply(get_med_name)


    def format_labs(self):
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

        self.labs = abnormal_labs


    def format_labs_continuous(self):
        labs = pd.read_csv(self.data_path + '5_lab_nor__lab_results_obr_p__lab_results_obx.csv')

        # Filter to applicable cols:
        labs2 = labs[
            ['lab_nor_enc_id', 'lab_results_obx_result_desc', 'lab_results_obx_observ_value', 'lab_results_obx_units']]

        # Drop NAs
        labs2.dropna(inplace=True)

        # Create new features
        labs2['lab_test'] = 'lab_' + labs['lab_results_obx_result_desc'] + ' (' + labs2['lab_results_obx_units'] + ')'
        labs2['lab_test_results'] = labs2['lab_results_obx_observ_value']
        labs2 = labs2[['lab_nor_enc_id', 'lab_test', 'lab_test_results']]

        def parse_results(x):
            try:
                x = float(x)
            except ValueError:  # strings
                if x in ['++POSITIVE++', 'POSITIVE', 'DETECTED']:
                    return 1

                if x in ['Negative', 'None Detected', 'None seen', 'Not Observed', 'NOT DETECTED',
                         'NEGATIVE', 'NEGATIVE CONFIRMED', '<20 NOT DETECTED', '<15 NOT DETECTED', '<1.30 NOT DETECTED',
                         '<1.18 NOT DETECTED', 'NONE DETECTED', '<1.0 NEG', 'NONE SEEN', '<1:10', '<1:16', '<1:64', ]:
                    return 0

                if x in ['Comment', 'FEW', 'INTERFERENCE', 'MANY', 'MODERATE', 'NOT APPLICABLE',
                         'NOT CALC', 'NOT CALCULATED', 'NOT GIVEN', 'NOTE', 'PACKED', 'PENDING', 'SEE BELOW',
                         'SEE NOTE', 'See Final Results', 'TNP', 'UNABLE TO CALCULATE', 'B']:
                    return None

                if x.startswith('> OR = '):
                    x = x.split('> OR = ')[1]

                if x.startswith('<'):
                    return float(x.split('<')[1]) - .1

                if x.startswith('>'):
                    return float(x.split('>')[1]) + .1

                if x.endswith('%'):  # drop percentages (should be in lab_test description)
                    return float(x.split('%')[0])

                if '-' in x:  # take average of range
                    x = x.split('-')
                    return (float(x[0]) + float(x[1])) / 2

                if x in ['6.9 % +', '6.9% +', '6.9%+']:
                    return 7

                if x == '7.5%+':
                    return 8

                if x == '9.8 % +':
                    return 10

                if ':' in x:  # take average of range
                    x = x.split(':')
                    x = (float(x[0]) + float(x[1])) / 2

            return float(x)

        labs2['lab_test_results'] = labs2['lab_test_results'].apply(lambda x: parse_results(x))
        labs2.dropna(inplace=True)
        labs2.rename(columns={'lab_nor_enc_id': 'enc_id'}, inplace=True)
        self.labs_cont = labs2


    # function to transform assessments table to merge with encounters
    def format_assessments(self):
        try:
            with open(self.data_path + 'bin/assessments.pickle', 'rb') as picklefile:
                self.assessments = pickle.load(picklefile)

        except FileNotFoundError:
            assessment = self.assessments
            # Filter for relevant enc_ids
            assessment = assessment[assessment['enc_id'].isin(
                set(assessment['enc_id']).intersection(
                    set(self.encounters.enc_id)))]

            # Collapse data down by enc_id for assessment
            assessment_text = assessment.groupby(['person_id', 'enc_id'])['txt_description'].apply(list)
            assessment_codeID = assessment.groupby(['person_id', 'enc_id'])['txt_diagnosis_code_id'].apply(list)
            assessment_text = pd.DataFrame(assessment_text)
            assessment_codeID = pd.DataFrame(assessment_codeID)

            # Merge series data from text and codeID columns into one df for assessment
            assessment2 = assessment_text.merge(assessment_codeID, how='left', on=['person_id', 'enc_id'])
            assessment2 = pd.DataFrame(assessment2)
            assessment2.reset_index(inplace=True)

            # Remove punctuation, convert all to lowercase, remove stopwords, tokenize, create bigrams

            # Remove Punctuation and convert to lower
            assessment2['txt_description'] = assessment2.txt_description.apply(
                lambda x: ', '.join([str(i) for i in x]))
            assessment2['txt_description'] = assessment2['txt_description'].str.replace('[^\w\s]', '')
            assessment2['txt_description'] = assessment2['txt_description'].str.lower()

            #remove words that are directly correlated to positive AD or Dementia
            ad_pos = 'alzheimers disease|dementia|late onset alzheimers|alzh|lewy|frontotemporal|hydrocephalus|huntington|wernicke|creutzfeldt|vascular dementia|Huntingtons disease|Mixed dementia|Parkisons disease'
            assessment2['txt_description'] = assessment2['txt_description'].str.replace(ad_pos,'')

            # tokenize
            assessment2['txt_tokenized'] = assessment2.apply(
                lambda row: nltk.word_tokenize(row['txt_description']), axis=1)

            # Remove Stopwords
            stop = stopwords.words('english')
            assessment2['txt_tokenized'] = assessment2['txt_tokenized'].apply(
                lambda x: [item for item in x if item not in stop])

            # Create ngrams
            assessment2['ngrams'] = assessment2.apply(
                lambda row: list(nltk.trigrams(row['txt_tokenized'])), axis=1)

            # Convert trigram lists to words joined by underscores
            assessment2['ngram2'] = assessment2.ngrams.apply(lambda row: ['_'.join(i) for i in row])

            # Convert trigram and token lists to strings
            assessment2['txt_tokenized2'] = assessment2['txt_tokenized'].apply(' '.join)
            assessment2['ngram2'] = assessment2.ngram2.apply(lambda x: ' '.join([str(i) for i in x]))
            assessment2['np_chunks'] = assessment2['txt_tokenized2'].apply(nlp_utils.getNounChunks)

            self.assessments = assessment2
            with open(self.data_path + 'bin/assessments.pickle', 'wb') as picklefile:
                pickle.dump(self.assessments, picklefile)


    def format_diagnosis(self):
        try:
            with open(self.data_path + 'bin/diagnosis.pickle', 'rb') as picklefile:
                self.diagnosis = pickle.load(picklefile)

        except FileNotFoundError:
            # Read in diagnosis table
            diagnosis = self.diagnosis
            ccsr = pd.read_csv(self.data_path + 'ccsr_mapping.csv')
            diagnosis['diagnosis_code_stripped'] = diagnosis['diagnosis_code_id'].str.replace(".", "")
            diagnosis = diagnosis.merge(ccsr, left_on='diagnosis_code_stripped', right_on='ICD-10-CM Code', how='left')

            diagnosis_icd9 = pd.DataFrame(diagnosis.groupby(['person_id', 'enc_id'])['icd9cm_code_id'].apply(list))
            diagnosis_dc = pd.DataFrame(diagnosis.groupby(['person_id', 'enc_id'])['diagnosis_code_id'].apply(list))
            diagnosis_desc = pd.DataFrame(diagnosis.groupby(['person_id', 'enc_id'])['description'].apply(list))
            diagnosis_datesymp = pd.DataFrame(diagnosis.groupby(['person_id', 'enc_id'])['date_onset_sympt'].apply(list))
            diagnosis_datediag = pd.DataFrame(diagnosis.groupby(['person_id', 'enc_id'])['date_diagnosed'].apply(list))
            diagnosis_dateresl = pd.DataFrame(diagnosis.groupby(['person_id', 'enc_id'])['date_resolved'].apply(list))
            diagnosis_statusid = pd.DataFrame(diagnosis.groupby(['person_id', 'enc_id'])['status_id'].apply(list))
            diagnosis_dx = pd.DataFrame(diagnosis.groupby(['person_id', 'enc_id'])['dx_priority'].apply(list))
            diagnosis_chronic = pd.DataFrame(diagnosis.groupby(['person_id', 'enc_id'])['chronic_ind'].apply(list))
            diagnosis_rcdelswhr = pd.DataFrame(
                diagnosis.groupby(['person_id', 'enc_id'])['recorded_elsewhere_ind'].apply(list))

            diagnosis_ccsr_category = pd.DataFrame(diagnosis.groupby(['person_id', 'enc_id'])['CCSR Category Description'].apply(list))
            # Merge series data from text and codeID columns into one df for assessment
            diagnosis2 = diagnosis_icd9 \
                .merge(diagnosis_dc, how='left', on=['person_id', 'enc_id']) \
                .merge(diagnosis_desc, how='left', on=['person_id', 'enc_id']) \
                .merge(diagnosis_datesymp, how='left', on=['person_id', 'enc_id']) \
                .merge(diagnosis_datediag, how='left', on=['person_id', 'enc_id']) \
                .merge(diagnosis_dateresl, how='left', on=['person_id', 'enc_id']) \
                .merge(diagnosis_statusid, how='left', on=['person_id', 'enc_id']) \
                .merge(diagnosis_dx, how='left', on=['person_id', 'enc_id']) \
                .merge(diagnosis_chronic, how='left', on=['person_id', 'enc_id']) \
                .merge(diagnosis_rcdelswhr, how='left', on=['person_id', 'enc_id']) \
                .merge(diagnosis_ccsr_category, how='left', on=['person_id', 'enc_id'])

            diagnosis2 = pd.DataFrame(diagnosis2)
            diagnosis2.reset_index(inplace=True)

            self.diagnosis = diagnosis2

            # drop 5 rows with missing diag data
            self.diagnosis = self.diagnosis[~self.diagnosis['diagnosis_code_id'].isnull()]

            with open(self.data_path + 'bin/diagnosis.pickle', 'wb') as picklefile:
                pickle.dump(self.diagnosis, picklefile)

    # return the main data output
    def create(self, name='dataloader'):
        print('Initializing DataLoader\n..loading CSVs')
        self.generate_csv_attributes()
        print('..loading meds')
        self.format_meds()
        print('..loading lab flags')
        self.format_labs()
        print('..loading continuous labs')
        self.format_labs_continuous()
        print('..loading diagnosis')
        self.format_diagnosis()
        print('..loading assessments')
        self.format_assessments()
        self.main = self.encounters.copy()

        # write to pickle file
        self.write(filename=name)
        print('Data load complete!\n')


    # helper function write entire class object to your project directory
    def write(self, filename='dataloader'):
        with open('bin/' + filename + '.pickle', 'wb') as picklefile:
            pickle.dump(self, picklefile)


    # helper function to return entire class object to your project directory
    def load(self, filename='dataloader'):
        try:
            with open('bin/' + filename + '.pickle', 'rb') as picklefile:
                return pickle.load(picklefile)
        except FileNotFoundError:
            print('DataLoader not found. Running create() to rebuild..')
            self.create()

if __name__ == "__main__":
    data = DataLoader(subset=1000)
    data.create('subset')
