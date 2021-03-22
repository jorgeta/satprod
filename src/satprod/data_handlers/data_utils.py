from datetime import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing

def datetime2path(datetime: datetime) -> str:
    '''
    Takes a datetime and converts it to the 
    corresponding path in string format.

    Input parameters:
        datetime: datetime variable

    Output parameters:
        path corresponding to the datetime
    '''
    
    return datetime.strftime('%Y/%m/%d/%H;%M;00.png')

def path2datetime(path: str) -> datetime:
    '''
    Takes string of the form "2018/03/20/03;15;00.png" and 
    converts it to the corresponding date in datetime format.

    Input parameters:
        path: of the form "/2018/03/20/03;15;00.png", (with a '/' in the beginning) is also handled
    
    Output parameters:
        datetime variable corresponding to the timepoint the image was taken
    '''

    if path[0]=='/': path = path[1:]
        
    assert path[-4:]=='.png', 'The path has to end with ".png"'
    return datetime.strptime(path[:-4], '%Y/%m/%d/%H;%M;%S')

def sin_transform(values):
    '''
    Takes the sine of each element in the numpy array values
    of degree values.

    Input parameters:
        values: numpy array of degree values

    Output parameters:
        values: numpy array of the sine of the input values, elementwise
    '''

    return np.sin(2*np.pi*values/360)

def cos_transform(values):
    '''
    Takes the cosine of each element in the numpy array values
    of degree values.

    Input parameters:
        values: numpy array of degree values

    Output parameters:
        values: numpy array of the cosine of the input values, elementwise
    '''

    return np.cos(2*np.pi*values/360)

def scaler(arr, min_val: float=0.0, max_val: float=1.0):
    '''
    Scales the input array so that it has the desired minimum and maximum value.

    Input parameters:
        arr: array for scaling
        min_val: minimum value of the returned scaled array
        max_val: maximum value of the returned scaled array
    
    Output parameters:
        scaled array
    '''
    
    return np.interp(arr, (np.min(arr), np.max(arr)), (min_val, max_val))

def max_min_scale_df(df: pd.DataFrame) -> pd.DataFrame:
    """[summary]

    Args:
        df (pd.DataFrame): dataframe or part of dataframe for scaling

    Returns:
        pd.DataFrame: scaled dataframe
    """
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, index = df.index, columns=df.columns)

def wind_degrees_to_polar(sin, cos):
    '''
    Converts sine and cosine of wind direction degrees into polar coordinate degrees.
    That is, 0 degrees means wind from west to east, 90 degrees means wind from
    south to nord, and so on.
    '''
    
    deg = np.zeros(len(cos))
    for i in range(len(cos)):
        if sin[i]>0:
            deg[i] = np.arcsin(cos[i])*180/np.pi
            if deg[i] < 0:
                deg[i] += 360
        elif cos[i]>0:
            deg[i] = np.arccos(sin[i])*180/np.pi
        else:
            deg[i] = np.arctan(cos[i]/sin[i])*180/np.pi+180
    return (deg+180) % 360

def MAE(arr1, arr2):
    return np.mean(np.abs(arr1-arr2))
    
def RMSE(arr1, arr2):
    return np.sqrt(np.mean(np.square(arr1-arr2)))

def get_columns(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    return df[[col for col in df.columns if keyword in col]]