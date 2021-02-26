import pandas as pd
from dask import dataframe as dd

base_url = "E:\\20201208_Dementia_AD_Research_David_Julovich\\QueryResult\\"

labs = pd.read_csv(base_url + "5_lab_nor__lab_results_obr_p__lab_results_obx.csv")


# BELOW IS READING IN ENCOUNTERS
def fetch_encounters():
    enct = pd.read_csv(base_url + "1_BaseEncounters_Dempgraphics_Payers.csv")

    enct.rename(columns={
        'EncounterDate': 'encounterdate',
        'Demographics': 'race',
        'Race': 'ethnicity',
        'Ethnicity': 'gender',
        'Gender': 'age',
        'AgeAtEnc': 'visittype',
        'VisitType': 'serviedepartment',
        'ServiceDepartment': 'locationname',
        'LocationName': 'reason_for_visit',
        'Reason_for_Visit': 'cpt_code'
    },
    inplace=True)
    return enct

# dave subsets columns here...i'm going to hold off on this for right now

# dave has some EDA here...not really for purpose of this specific doc.
# can reference the original for EDA graphs


# BELOW IS READING IN CPT CODE
def fetch_cpt():
    cpt = pd.read_csv(base_url + "2_CPT_Codes.csv")
    cpt.rename(columns={'CPT_Code': 'code'}, inplace=True)

    # create dummies of original
    cpt_wide = pd.get_dummies(cpt, columns=['code'], drop_first=True)
    cpt_wide = cpt_wide.iloc[:, 2:]  # subsetting columns
    # renaming columns in cpt to add a "cpt" prefix
    cpt_wide.columns = ['cpt_' + col for col in cpt_wide.columns]

    # rejoining encounter id back onto the wide dataset
    cpt_wide = pd.concat([cpt['enc_id'], cpt_wide], axis=1)

    # make sure we retain a one-one relationship with encounters.
    # grouping by encounter in case one person can have multiple cpt codes
    cpt_wide = cpt_wide.groupby(by='enc_id', as_index=False).max()
    return cpt_wide
# below here, David drops columns...again, let's wait until we have total product to do this.

# now we need to rejoin the cpt table back onto the encounter table
#main = enct.merge(cpt_wide, on='enc_id')


# ICD table
def fetch_icd():
    icd = dd.read_csv(base_url + '6_patient_diagnoses.csv')
    icd = icd.categorize()
    icd_wide = dd.get_dummies(icd, columns=['diagnosis_code_id'])
    print(len(icd_wide.index))
    enc_list = icd_wide['enc_id'].unique()
    output = []
    x = 1
    total = len(enc_list)
    for enc in enc_list:
        print("{}% complete".format(round(x/total)))
        output.append(icd_wide[icd_wide['enc_id'] == enc]\
                      .groupby('enc_id').max().compute())
        x += 1
    return pd.concat(output, axis=0)


# merging main table with icd table
#main = main.merge(icd_wide, on='enc_id')
#vitals = pd.read_csv(base_url + '3_vitals_signs.csv')

if __name__ == "__main__":
    fetch_icd()
    input('')


