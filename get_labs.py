## RUN LOAD_DATA FIRST

# %%
def get_labs(encoded=False):
    """
    returns the labs dataframe consisting of person_id, enc_id and lab_results
    encoded: returns one-hot encoded dataframe if True, otherwise returns a set
    """

    data_path = 'E:/20201208_Dementia_AD_Research_David_Julovich/QueryResult/'
    labs = pd.read_csv(data_path + '5_lab_nor__lab_results_obr_p__lab_results_obx.csv')


    # %% Remove Deleted rows and drop deleted indicators
    labs = labs[labs['lab_nor_delete_ind']=='N']
    labs = labs.drop(columns=['lab_nor_delete_ind', 'lab_results_obx_delete_ind'])

    # Remove incomplete labs & drop column
    labs = labs[labs['lab_nor_completed_ind']=='Y']
    labs = labs.drop(columns=['lab_nor_completed_ind'])

    # Remove pending labs
    labs = labs[labs['lab_nor_test_status']!='InProcessUnspecified']
    labs = labs[labs['lab_nor_test_status']!='Pending']

    # Find abnormal tests:
    abnormal_indicators = ['L', 'H', 'A', '>', 'LL', 'HH', '<']

    # Set lab_results to result description if exists, otherwise use test description
    labs['lab_results'] = np.where(
        labs['lab_results_obx_result_desc'].notnull(), 
        labs['lab_results_obx_result_desc'], 
        labs['lab_results_obr_p_test_desc']
        )

    # Combine outcome with test result
    labs['lab_results'] = np.select(
        [ #condition
            labs.lab_results_obx_abnorm_flags == '>',
            labs.lab_results_obx_abnorm_flags == 'H',
            labs.lab_results_obx_abnorm_flags == 'HH', 
            labs.lab_results_obx_abnorm_flags == '<', 
            labs.lab_results_obx_abnorm_flags == 'L', 
            labs.lab_results_obx_abnorm_flags == 'LL', 
            labs.lab_results_obx_abnorm_flags == 'A'        
            ],
        [ #value
            'HIGH ' + labs['lab_results'],
            'HIGH ' + labs['lab_results'],
            'VERY HIGH ' + labs['lab_results'],
            'LOW ' + labs['lab_results'],
            'LOW ' + labs['lab_results'],
            'VERY LOW ' + labs['lab_results'],
            'ABNORMAL ' +labs['lab_results']
            ],
        default = 'NORMAL'
    )

    labs[labs['lab_results'].notnull()]

    # Capture abnormal labs
    abnormal_labs = labs[labs['lab_results'] != 'NORMAL']
    abnormal_labs['lab_results'] = abnormal_labs['lab_results'].str.title()
    abnormal_labs = pd.DataFrame(abnormal_labs[['lab_nor_person_id','lab_nor_enc_id','lab_results']].groupby(
        ['lab_nor_person_id','lab_nor_enc_id'])['lab_results'].apply(set))

    abnormal_labs.reset_index(inplace=True)    
    abnormal_labs.columns = ['person_id','enc_id','lab_results']

    if encoded == False:
        return abnormal_labs
    
    # Pandas get_dummies function will not parse lists, enter the multiLabelBinarizer 
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    encoded_labs = abnormal_labs.join(pd.DataFrame(mlb.fit_transform(abnormal_labs['lab_results']),columns=mlb.classes_))
    
    return encoded_labs
