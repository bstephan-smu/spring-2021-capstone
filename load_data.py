# %% Requirements:
import pandas as pd
import numpy as np
import re
import os


class DataLoader:
    def __init__(self, data_path='E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/'):
        # Load data: storing as environment variable in case sharing across multiple files
        os.environ['data_path'] = data_path
        self.data_path = data_path

        # regex phrase to lookup alzheimers diagnosis
        self.alz_regex = 'alzh'

        # store list of dementia icd codes here...will be updated in later function
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

        # reading in raw data from all datasets
        self.encounters = pd.read_csv(self.data_path + '1_BaseEncounters_Dempgraphics_Payers.csv')
        self.format_encounters()  # align encounters table columns...make sure columns are aligned as planned

        self.cpt = pd.read_csv(self.data_path + '2_CPT_Codes.csv')
        self.vitals = pd.read_csv(self.data_path + '3_vitals_signs.csv')
        self.meds = pd.read_csv(self.data_path + '4_patient_medication.csv')
        self.labs = pd.read_csv(self.data_path + '5_lab_nor__lab_results_obr_p__lab_results_obx.csv')
        self.diagnosis = pd.read_csv(self.data_path + '6_patient_diagnoses.csv')
        self.assessments = pd.read_csv(self.data_path + '7_assessment_impression_plan_.csv')

    # Fix header on encounters:
    def format_encounters(self):
        self.align_columns()  # ensure that encounters columns are properly aligned
        self.encounters['EncounterDate'] = pd.to_datetime(self.encounters['EncounterDate'], format='%Y%m%d')
        self.encounters = self.encounters.head(1000)  # making a smaller sample of data for speed in dev environment

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

    # return the main data output
    def create(self):
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
        meds_wide = pd.get_dummies(self.meds[['enc_id', 'medid', 'is_currently_taking']]
                                   .query('is_currently_taking'), columns=['medid']) \
            .groupby('enc_id', as_index=False).max()

        main = main.merge(meds_wide, on='enc_id')

        # step 5...load labs onto the main dataframe
        # come back to this one dustin is currently on it

        # step 6...load diagnosis onto main dataframe
        # come back to this one...jeff is currently working on this guy

        # step 7...load assessments onto main dataframe
        # double check to see if jeff has converted this portion yet.

        return main

    # helper function write main dataframe
    def write(self):
        pass

    # helper function to return main dataframe
    def load(self):
        pass


if __name__ == "__main__":
    data = DataLoader()
    main = data.create()
    print(main.head())
