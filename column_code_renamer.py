
# %%
import pandas as pd
from load_data import DataLoader
from numpy.core import shape
data=DataLoader().load()
# %%
data.main
# %%
def drop_cpt(self ,prefix): 
    self.columns = self.columns.str.lstrip(prefix)
    file=pd.read_csv('E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/cpt_updated_full.csv')
    f_list=file['cpt'].values.tolist()
    old_cpt=f_list
    file=pd.read_csv('E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/cpt_updated_full.csv')
    f_list=file['short_description'].values.tolist()
    new_cpt=f_list
    res_cpt = dict(zip(old_cpt,new_cpt))
    self.rename(columns=res_cpt, inplace=True)

    
#%%     
def drop_icd(self ,prefix): 
    self.columns = self.columns.str.lstrip(prefix)
    file=pd.read_csv('E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/icd_updated_full.csv')
    f_list=file['ICD_CODE'].values.tolist()
    old_icd=f_list
    file=pd.read_csv('E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/icd_updated_full.csv')
    f_list=file['SHORT_DESCRIPTION'].values.tolist()
    new_icd=f_list
    res_icd = dict(zip(old_icd,new_icd))
    self.rename(columns=res_icd, inplace=True)

# %%
drop_cpt(data.main, 'cpt_')
drop_icd(data.main, 'asmt_icd_')
# %%
for col in data.main.columns: 
    print(col)

