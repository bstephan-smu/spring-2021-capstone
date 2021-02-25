

#meds<-read_csv("//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/4_patient_medication.csv")
labs<-read_csv("//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/5_lab_nor__lab_results_obr_p__lab_results_obx.csv")

#asses<-read.csv("//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/7_assessment_impression_plan_.csv")
library("readr")
library(reshape2)
library(plyr)
library(skimr)
library(dplyr)
library(ggplot2)


###################ENCOUNTER TABLE###########################################################
enct<-read_csv("//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/1_BaseEncounters_Dempgraphics_Payers.csv")

# align the column names to the correct data and clean up column names
enct %>%rename(encounterdate=EncounterDate, race=Demographics,ethnicity=Race,gender=Ethnicity,age=Gender,visittype=AgeAtEnc,serviedepartment=VisitType,locationname=ServiceDepartment,reason_for_vist=LocationName, cpt_code=Reason_for_Visit)->enct
#remove useless columns
enct <- enct[ -c(3,4,11,12,15:19) ]

#add enc_ prefix to all columns for encounters tablw
colnames(enct) <- paste("enc", colnames(enct), sep = "_")

skim(enct)

#maybe the nulls should be dropped, there appears to be a trend for NULLS
sum(enct$enc_gender == "NULL")
sum(enct$enc_race == "NULL")
sum(enct$enc_ethnicity == "NULL")

# NA and NULLS dropped for gender and age
no_na_enct<-enct[!(enct$enc_gender=="NULL"),]
sum(no_na_enct$enc_gender == "NULL")
sum(no_na_enct$enc_age == "NULL")


#count the number patient ids 
#we find 5240 unique patient ids with nulls and 3914 w/out nulls 
count_person_ids<-as.data.frame(table(enct$enc_person_id))

#we see replicates of from 1 to 111 this means there is a signal patient 
#that has 111 encounters
skim(count_person_ids)

#to confirm unique encounter we count the unique enc_ids and find 50650 unique encounter ids
#since all encounters are unique
#this is where all merging will stem because we have complete uniquness of records 
#table also included the sex, age ,gender required for modeling
count_enct_ids<-as.data.frame(table(enct$enc_enc_id))

#Plotting distribution of encounters by the number of times a unique patient Id appeared
ggplot(count_person_ids, aes(x=Freq))+geom_histogram(color="black", fill="white",binwidth = 1)+
  ylab("Count of Patients") + xlab("Number of Visits")+ylim(-10,750)

#Here we see the box plot of the frequency of encounter bas on unique ids
#The plot has 75% of the patient population having number of encounters ranging from 3 to 14
#with the mean count of encounters at 9.6 
mean<-mean(count_enct_ids$Freq)
ggplot(count_person_ids, aes(y=Freq))+
  geom_boxplot( ) +annotate("point", x = 0, y = 9.666, colour = "blue", size =3)



#writing two clean and confirmed encounter table 
#on with nulls for gender and age and one without nulls
write.csv(enct,"//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/clean_tables/enct.csv") 
write.csv(no_na_enct,"//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/clean_tables/no_na_enct.csv") 


########REMEMBER TO SCALE AND NORMALIZE DATA FOR MODELING

############################################################################################



####################CPT CODE TABLE#########################################################
cpt<-read_csv("//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/2_CPT_Codes.csv")


#clean up column names in cpt table
cpt%>% rename(cpt_code= CPT_Code)->cpt

#add cpt_ prefix to all columns for CPT table
colnames(cpt) <- paste("cpt", colnames(cpt), sep = "_")

#Drop useless colums
cpt <- cpt[ -c(3)]

#Count the unique of enc_id in the CPT table
#We find 50738  this is very close the number of enc_enc_id.  So we will join
#the cpt and encounter table by enc_id.  The extra observation are useless since they 
#cannot be linked to patient.  
count_cpt_enct_ids<-as.data.frame(table(cpt$cpt_enc_id))


#reshape cpt to sparce matrix
cpt_wide<-dcast(cpt, cpt_enc_id ~ cpt_cpt_code,value.var ="cpt_cpt_code")


#the count the of enc_ids in the wide table equals the count in the orginal cpt table
#make list of UNIQUE cpt codes, we find 297 which equal the cpt_wide just minus cpt_enc_id column
#Many NAs in this list need to find descriptions
count_cpt_codes<-as.data.frame(table(cpt$cpt_cpt_code))

count_cpt_codes<-count_cpt_codes[order(-count_cpt_codes$Freq),]

#plot the frequency distribution of 297 different cpt codes.   
ggplot(count_cpt_codes, aes(x=Freq, y=Var1))+geom_histogram(stat = "identity")+xlim(0,2500)
  

write.csv(count_cpt_codes,"//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/cpt_code_list.csv")

write.csv(cpt_wide,"//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/clean_tables/cpt_wide.csv")

##################################################################################################



################MERGE THE ENCOUNTER AND CPT TABLES#############################################
e_c_table<-merge(enct, cpt_wide, by.x = 'enc_enc_id', by.y = 'cpt_enc_id' )
#count of the encounter ids in merged table matched the count from the encounter table.   

#5240 patient are maintained after merge
count_ids_e_c<-as.data.frame(table(e_c_table$enc_person_id))

#sanity check for #'s of columns  
10+298-1



write.csv(e_c_table,"//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/clean_tables/e_c_table.csv")

#merge encounter and cpt with no nas and number of unique encounters
#is droppend to 43953  THIS MAY BE WHAT WE SHOULD USE
NoNA_e_c_table<-merge(no_na_enct, cpt_wide, by.x = 'enc_enc_id', by.y = 'cpt_enc_id' )
 
#The unique person ids drop to 3914 using no nas 
NoNa_count_person<-as.data.frame(table(NoNA_e_c_table$enc_person_id))

write.csv(NoNA_e_c_table,"//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/clean_tables/NoNA_e_c_table.csv")


#############################################################################################



##################################ICD TABLE################################################
icd<-read_csv("//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/6_patient_diagnoses.csv")

#look at NA and useless columns
# At least 3 columns are almost all NAs
skim(icd)



#add cpt_ prefix to all columns for CPT table
colnames(icd) <- paste("icd", colnames(icd), sep = "_")

all(icd$icd_icd9cm_code_id == icd$icd_diagnosis_code_id)


sum(icd$icd_status_id == "NULL")
sum(icd$icd_dx_priority == "NULL")
sum(icd$icd_chronic_ind == "N")
sum(icd$icd_recorded_elsewhere_ind== "N")
#drop useless columns
icd<- icd[ -c(3,6:12) ]

#Count the unique of enc_id in the ICD table
#We find 46690 unique encounters this is less than the numbers  of 
#in the encounters and cpt tables, it is what it is 
count_icd_enc_ids<-as.data.frame(table(icd$icd_enc_id)) 

#Count the unique of person_id in the ICD table
#We find 4995 this is less than the encounter and cpt tables, it is what it is
count_icd_person<-as.data.frame(table(icd$icd_person_id))

#We find  4753 of unique icd codes for all 4995 patients
count_icd_codes<-as.data.frame(table(icd$icd_diagnosis_code_id))

#drop the icd_pat column for merge
icd<- icd[ -c(1) ]

#reshape icd to sparce matrix
icd_wide<-reshape2::dcast(icd, icd_enc_id ~icd_diagnosis_code_id, value.var ="icd_diagnosis_code_id")


#check the count of icd_enc_id after reshape 46690 same as icd 
count_icd_wide_enc_id<-as.data.frame(table(icd_wide$icd_enc_id))

#count_wide_codes equals the original number of codes in icd table
write.csv(icd_wide,"//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/icd_wide.csv")

############################################################################################



###################MERGE E_C_M TABLE WITH ICD TABLE###########################################

#when merging e_c table to icd we maintain 46690 unique enc_ids
e_c_i_table<-merge(e_c_table, icd_wide, by.x = 'enc_enc_id', by.y = 'icd_enc_id' )


#4995 patient are maintained after merge
count_ids_e_c_i<-as.data.frame(table(e_c_i_table$enc_person_id))

#sanity check for # of columns  
10+298+4754-2

#when merging on_nas_e_c table to icd drop to 40695  unique enc_ids
NoNA_e_c_i_table<-merge(NoNA_e_c_table, icd_wide, by.x = 'enc_enc_id', by.y = 'icd_enc_id' )

#the count of unique person ids drop to 3814 when using no nas
NoNa_count_person<-as.data.frame(table(NoNA_e_c_i_table$enc_person_id))
#############################################################################################



#######################VITALS TABLE########################################################
library(readr)

vital<-read_csv("//hsc.ad.unt.edu/DATA/IntMedHABS/Wet-Lab/David/AD_CAPA/3_vitals_signs.csv")

#add cpt_ prefix to all columns for Vital table
colnames(vital) <- paste("vit", colnames(vital), sep = "_")

#Drop useless columns
vital <- vital[ -c(4,5,7:10,13:16,18,20,21,25,27,28,30,31)]

#Count the unique of enc_id in the ICD table
#We find 45283 this is less than the numbers in the encounters, cpt and icd tables
count_vital_enc<-as.data.frame(table(vital$vit_enc_id))
#skim(count_vital_enc)


#We find 52184 is than the numbers in the encounters for this table so we cleanout 
#duplicate or nonsense sequencse
count_vital_seq<-as.data.frame(table(vital$vit_seq_no))

#We also find 4991 this slightly is less than the numbers in the encounters, cpt and icd tables
count_vital_person<-as.data.frame(table(vital$vit_person_id))


gb<-vital %>% group_by(vit_seq_no)

dups<- vital %>% filter_at(vars(-3), all_vars(duplicated(.)))

vital %>% filter_at(vars(vit_enc_id), all_vars(. == .vital$vit_seq_no))->dup2

vital %>%  mutate_at(vars(-2)) %>%
  filter_at(vars(-2), all_vars(. == vital$vit_seq_no))->dup2

dup<-vital$vit_enc_id[duplicated(vital$vit_enc_id)]

dups<-dups[order(dups$vit_enc_id),]



#############################################################################################


#############################MERGE E_C_I_M TABLE WITH VITALS TABLES#########################



##########################ASSESMENT TABLE#####################################################
#