from datetime import datetime
import numpy as np

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
    return np.interp(arr, (np.min(arr), np.max(arr)), (min_val, max_val))
