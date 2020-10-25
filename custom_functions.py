
# function for seperating out desired features
def slice_feats(dataframe, feat_names):
    """This function takes in a dataframe and a list of the desired feature's names.
    It then slices those features out of the dataframe and returns the dataframes X and y values"""
    import pandas as pd
    
    sliced_dataframe = dataframe[feat_names + ["Class"]]
    
    return sliced_dataframe.drop("Class", axis=1), sliced_dataframe.Class
    
    
    

# Function for resampling
def resample_data(X, y):
    """This function takes in the X and y values for the training set.
    Then it resamples the data. Finally it returns the new
    X and y values"""
    
    from imblearn.under_sampling import NearMiss
    
    nm = NearMiss()
    
    return nm.fit_resample(X, y)

def slice_resample_data(dataframe, feat_names):
    """This function takes in a dataframe and a list of the desired feature's names. 
    It then slices the correct features out of the dataframe and splits it into X and y 
    variables. Then it resamples the data using the NearMiss algorithm. Finally it returns 
    the resampled data as X and y variables"""
    
    X, y = slice_feats(dataframe, feat_names)
    
    return resample_data(X, y)