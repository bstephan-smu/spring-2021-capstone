# %% DO NOT RUN!!!
# 2/10/21 PATCH_1: fix for failed loader during merge_assessments
# this just runs merge_assessments manually, with a patch to handle NAs inside the CCSR lists 
from load_data import DataLoader
capData = DataLoader().load('partial')

assess_copy = capData.asmt_diag.copy()

# Pair down assessments table to columns of interest
assess_copy = assess_copy[
    ['person_id', 'enc_id', 'txt_description', 'txt_tokenized', 'ngrams', 'ngram2', 'txt_tokenized2',
        'np_chunks', 'CCSR Category', 'description']]



def one_hot(df, col_name, prefix=''):
    mlb = MultiLabelBinarizer()
    df = df.join(pd.DataFrame(mlb.fit_transform(df[col_name]), columns=prefix + mlb.classes_))
    df = df.drop(columns=[col_name])
    return df


def stripNA(lst_col):
    for item in lst_col:
        if type(item) != str:
            lst_col.remove(item)
    return lst_col

assess_copy['CCSR Category2'] = assess_copy['CCSR Category'].apply(stripNA)
assess_encoded = one_hot(assess_copy, 'CCSR Category2', prefix='ccsr_')

assess_encoded.rename(columns={
    'txt_description': 'asmt_txt_description',
    'txt_tokenized': 'asmt_txt_tokenized',
    'ngrams': 'asmt_ngrams',
    'ngram2': 'asmt_ngram2',
    'txt_tokenized2': 'asmt_txt_tokenized2',
    'np_chunks': 'asmt_np_chunks',	
    'description': 'asmt_description'
}, inplace=True)

capData.main.drop(columns=['CCSR Category'], inplace = True)

capData.main = capData.main.merge(assess_encoded, on=['person_id', 'enc_id'], how='left')

capData.encode_encounters()

capData.clean()

capData.write()


# %% Patch 1.1: add clusters

from load_data import DataLoader
capData = DataLoader().load()

capData.merge_clusters()

capData.write()

# %%