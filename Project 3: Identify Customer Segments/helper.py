"""
All the functions for Identify Customer Segments
"""
def string_to_list(string):
    """convert string into a list of integers and strings"""
    list_of_strings = (string.replace('[', '').replace(']', '').split(','))
    for i in range(len(list_of_strings)):
        if list_of_strings[i] in ['-1', '0', '9']:
            list_of_strings[i] = int(list_of_strings[i])
    return list_of_strings

def convert_values_to_nan(df, features_df):
    """Convert values in Udacity_AZDIAS_Subset.csv that correspond to the list 
    'missing_or_unknown' in AZDIAS_Feature_Summary.csv
    """
    # Convert features_df['missing_or_unknown'] from a string to a list
    features_df['nulls'] = features_df['missing_or_unknown'].map(string_to_list)
    
    # Convert null values in df to np.nan
    for col in list(df.columns):
        list_of_nulls = list(features_df.loc[features_df.attribute==str(col)]['nulls'])[0]
        df.loc[df[col].isin(list_of_nulls),col] = np.nan
        
    return df

def get_null_percentage(df):
    """
    Input: dataframe
    Output: series with percentage of null values for each column
    """
    null_perc = df.isnull().mean()*100
    null_perc = null_perc.sort_values(ascending=False)
    return null_perc

def get_outlier_columns(df, threshold):
    """
    Returns a list of columns that have a percentage of null values
    Above threshold
    """
    null_perc = get_null_percentage(df)
    outlier_columns = null_perc[null_perc > threshold].index
    
    return outlier_columns
    
def remove_outlier_columns(df, outlier_columns, verbose = False):
    """
    Removes columns from a dataframe that have a percentage of null values
    Above a certain percentage threshold
    The null percentage of of the columns is 
    """
    columns_removed = []
    for col in outlier_columns:
        if col in df.columns:
            df.drop([col], axis = 1, inplace = True)
            num_columns_removed += 1
            columns_removed.append(col)
    if verbose:
        print('{} columns removed from dataframe'.format(num_columns_removed))
        print('{}'.format(list(columns_removed)))
    return df

def get_percent_null_per_row(df):
    """
    Returns a series 'percent_null' containing
    percentage of columns in row with null values
    """
    percent_null = df.isnull().sum(axis=1)/len(df.columns)*100
    
    return percent_null

def divide_dataframe(df, col_name, threshold):
    df_below = df.loc[df[col_name] <= threshold]
    df_above = df.loc[df[col_name] > threshold]
    
    return df_below, df_above

def plot_comparison(df_b, df_a, cols):
    n_col = len(cols)
    plt.figure(figsize=(15, 5*n_col))
    row=0
    for col in cols:
        plt.subplot(n_col, 2, row+1)
        sns.countplot(df_b[col])
        if row == 0:
            plt.title('Below Threshold')
        plt.subplot(n_col, 2, row+2)
        sns.countplot(df_a[col])
        if row == 0:
            plt.title('Above Threshold')
        row +=2
    plt.show()
    
def get_numeric_features(df, feature_df):
    ordinal = list(feature_df.loc[feature_df['type'] == 'ordinal']['attribute'])
    numeric = list(feature_df.loc[feature_df['type'] == 'numeric']['attribute'])
    interval = list(feature_df.loc[feature_df['type'] == 'interval']['attribute'])
    numeric_features = ordinal + numeric + interval
    
    return numeric_features

def get_object_features(df):
    object_features = list(df.select_dtypes(include=['object']).columns)
            
    return object_features

def get_categorical_features(df, feature_df):
    """
    Returns categorical features 
    Excludes objects
    All categories are already numerically encoded by design
    """
    # All mixed features
    mixed = list(feature_df.loc[feature_df['type'] == 'mixed']['attribute'])
    # All categorical features
    categorical = list(feature_df.loc[feature_df['type'] == 'categorical']['attribute'])
    combined = mixed + categorical
    categorical_features = list(set(combined) - set(get_object_features(df)))
                        
    return categorical_features

def clean_nulls(df, feature_df, fill_value=0, verbose=False):
    clean_df = df.copy()
    numeric = get_numeric_features(df, feature_df)
    objects = get_object_features(df)
    categoric = get_categorical_features(df, feature_df)
    
    # replace nan with fill value
    for col in [numeric, categoric]:
        clean_df[col] = clean_df[col].fillna(fill_value)
    
    clean_df[objects] = clean_df[objects].fillna(str(fill_value))
    
    # Need to get rid of negative numbers for one hot encoding
    clean_df[categoric] = clean_df[categoric].replace(to_replace=-1, value=fill_value)

    if verbose:
        print("Before Cleaning: {}".format(df.isnull().sum().sum()))
        print("After Cleaning: {}".format(clean_df.isnull().sum().sum()))
        
    return clean_df

def encode_objects(df):
    '''
    Convert string categories into integers
    fits and transforms
    only use on training set for now
    '''
    global le_OST_WEST_KZ
    le_OST_WEST_KZ = LabelEncoder()
    df['OST_WEST_KZ'] = le_OST_WEST_KZ.fit_transform(df['OST_WEST_KZ'])
    
    global le_CAMEO_DEUG_2015
    le_CAMEO_DEUG_2015 = LabelEncoder()
    df['CAMEO_DEUG_2015'] = le_CAMEO_DEUG_2015.fit_transform(df['CAMEO_DEUG_2015'])
    
    global le_CAMEO_DEU_2015
    le_CAMEO_DEU_2015 = LabelEncoder()
    df['CAMEO_DEU_2015'] = le_CAMEO_DEU_2015.fit_transform(df['CAMEO_DEU_2015'])
    
    global le_CAMEO_INTL_2015
    le_CAMEO_INTL_2015 = LabelEncoder()
    df['CAMEO_INTL_2015'] = le_CAMEO_INTL_2015.fit_transform(df['CAMEO_INTL_2015'])
    return df

def one_hot_encode_categories(df, cat):
    """
    One hot encodes arrays of integers
    Returns an array of the encoded features
    """
    global ohe
    ohe = OneHotEncoder()
    df_ohe = ohe.fit_transform(df[cat])
    
    return df_ohe

def convert_to_dummies(df, clowder):
    """
    Makes dummies
    Returns a dataframe with nice columns
    """
    dummies = pd.DataFrame()
    for cat in clowder:
        dummy = pd.get_dummies(df[cat], prefix = cat)
        dummies = pd.concat([dummies, dummy], axis = 1)
    
    return dummies

def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    
    
    # remove selected columns and rows, ...

    
    # select, re-encode, and engineer column values.

    
    # Return the cleaned dataframe.