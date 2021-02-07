# %%

from load_data import DataLoader
from numpy.core import shape
data=DataLoader().load()
# %%
data.main
# %%

# %%

def drop_cpt(self ,prefix):   
    #strip prefix and make a list of col names    
    self.columns = self.columns.str.lstrip(prefix)
    old_n=self.columns.tolist()
    #Read in the cpt desct_2 file and makea list of new cols name.  
    #since we one hot encore the order of header to list remained the same 
    file=pd.read_csv('E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/cpt_descriptions_2.csv')
    new_n=file['short_description'].values.tolist()
    #rename the columns of main
    self.rename(columns=dict(zip(old_n, new_n)),  inplace=True)
    #drop nan labeled columns
    self= self.drop["nan"]
    
def drop_icd(self ,prefix):   
    #strip prefix and make a list of col names    
    self.columns = self.columns.str.lstrip(prefix)
    old_n=self.columns.tolist()
    #Read in the cpt desct_2 file and makea list of new cols name.  
    #since we one hot encore the order of header to list remained the same 
    file=pd.read_csv('E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/icd_descriptions_2.csv')
    new_n=file['short_description'].values.tolist()
    #rename the columns of main
    self.rename(columns=dict(zip(old_n, new_n)), inplace=True)
    #drop nan labeled columns
    self= self.drop["nan"]

# %%
drop(data.main,'cpt_')

# %%
data.main 
