import pandas as pd
import numpy as np
import json

class DataLoader:
    """_summary_
    input : csv file
    output : Series of Field object

    Field's attribute : address, coordinate, Area, (polygon)
    """
    
    def __init__(self, file_name):
        self.dataset = self.load_data(pd.read_csv(file_name))
        
        
    def load_data(self, raw_dataset):
        """ 주소명, 좌표 튜플, 면적, polygon만 추출해야 함
        """
        dataset = raw_dataset.iloc[:,0 : 5]
        dataset.rename(columns = {'Column1':'address','Column2':'area','Column3':'x','Column4':'y','Column5':'polygon'}, inplace=True)
        dataset['coordinate']= list(zip(dataset['x'],dataset['y']))
        return dataset
    
    def convert_to_coordes(self):
        coord_list = self.dataset['coordinate']
        coord_nparray = np.array([[float(x), float(y)] for x, y in coord_list])
        return coord_nparray   
    