#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pdpipe as pdp
pd.set_option('display.max_columns', 1000)


# # Base Encounters
# Below is a dataframe of all visits (encounters)

# In[2]:


df_base_encounters = pd.read_csv("E:\\20201208_Dementia_AD_Research_David_Julovich\\QueryResult\\1_BaseEncounters_Dempgraphics_Payers.csv")
df_base_encounters.head()


# In[3]:


df_base_encounters.drop(columns=['Encounter_Teritiary_Payer'], inplace=True)
df_base_encounters.columns = ['person_id', 'enc_id', 'place_of_service', 'Provider_id', 'EncounterDate', 'Race',
                              'Ethnicity', 'Gender', 'AgeAtEnc', 'VisitType', 'ServiceDepartment', 'LocationName',
                             'Reason_for_Visit', 'CPT_Code', 'CPT_Code_Seq', 'Encounter_Primary_payer',
                              'Encounter_Secondary_Payer','Encounter_Teritiary_Payer']


# In[4]:


df_base_encounters.head()


# In[5]:


df_base_encounters['EncounterDate'] = pd.to_datetime(df_base_encounters['EncounterDate'], format='%Y%m%d')


# # CPT Codes

# In[6]:


cpt_codes = pd.read_csv("E:\\20201208_Dementia_AD_Research_David_Julovich\\QueryResult\\2_CPT_Codes.csv")
cpt_codes.head()
