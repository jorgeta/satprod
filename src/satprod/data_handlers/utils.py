from datetime import datetime
import numpy as np

def datetime2path(datetime: datetime) -> str:
    '''Takes a datetime and converts it to the 
    corresponding path in string format'''
    
    return datetime.strftime('%Y/%m/%d/%H;00;00.png')

def path2datetime(path: str) -> datetime:
    '''Takes string of the form "2018/03/20/00;00;00.png" and 
    converts it to the corresponding date in datetime format.
    
    Inputs of the form "/2018/03/20/00;00;00.png", is also handled'''
    if path[0]=='/':
        path = path[1:]
        
    assert path[-4:]=='.png', 'The path has to end with ".png"'
    return datetime.strptime(path[:-4], '%Y/%m/%d/%H;%M;%S')

def sin_transform(values):
    return np.sin(2*np.pi*values/360)

def cos_transform(values):
    return np.cos(2*np.pi*values/360)