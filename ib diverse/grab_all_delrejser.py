# -*- coding: utf-8 -*-
"""
Spyder Editor

To grab all delrejser into pandas dataframes from zipped .csv files and write them in feather files

This allows fast acces to the original delrejse data

the original data can then be selected and filtered from the feather files. 

pyarrow should be installed for this module to work

"""
import pandas as pd
import zipfile 
import os
from pathlib import Path
import datetime

WRITE_PATH = Path(r'data/')
READ_PATH =Path(r'//tsdw03/h/Data/Rejsedata/2019/Delrejser/')



# zfile = Path(r'//tsdw03/h/Data/Rejsedata/2019/Delrejser/delrejse_2019_12_12.zip')
# zfile = Path(r'data/delrejse_2019_12_12.zip')

def csv_grab_one_write(zfile,write_path,nrows = None): 
    '''
    takes a csv file with delrejse informations 
    
    writes it as a pandas dataframe to a feather file (.ft)
    
    datatypes are managed to save space 

    Parameters
    ----------
    zfile : pathlib.Path
        File to read.
    write_path : pathlib.Path
        Path to write the feather file .
    nrows : TYPE, optional
        number of rows if thesting . The default is all lines 

    Returns
    -------
    None.

    '''
    nrows_here = 200_000_000_000 if type(nrows) == type(None) else nrows
    print(f'Reading: {zfile}')
    # to get a feel of the data 
    df_probe = pd.read_csv(zfile,encoding='iso-8859-1',nrows  = 100 )
    relevant_columns = [c for c in df_probe.columns if c.upper()] # to get all columns 
    
    # manage the datatypes 
    dtype_dict0 = {'turngl' : 'int64' , 'ContractorId':'category',
                   'RejsePris':'float32','tidsrabat':'float32',
                   'ZonerRejst' : 'category' ,
                   'Applicationtransactionsequencenu' : 'int64',
                   'KortnrKrypt'  : 'string'
                   }
    
    dtype_dict =  {v:dtype_dict0.get(v,'category') for v in relevant_columns if v != 'Msgreportdate' }
    
    # now read 
  
    df = pd.read_csv(zfile,encoding='iso-8859-1',nrows = nrows_here  ,usecols = relevant_columns,
                     parse_dates = ['Msgreportdate'],dtype=dtype_dict,decimal=',',low_memory = True,
                     thousands='.' )
    
    write_path.mkdir(parents=True, exist_ok=True)
    ffile = (write_path /zfile.name).with_suffix('.ft')
    print(f'Writing {len(df)} lines to :{ffile} ')
    with open(ffile,'bw') as f:
        df.to_feather(f)
        
        
def csv_grab_all_write(read_path=READ_PATH,write_path=WRITE_PATH,nrows=2_000):
    '''Takes all delrejse*.zip files from read_path and writes to write_path,
    only nrows are read. this is to enable testing on smaller data sampels.
    '''
    for zfile in read_path.glob('delrejse*.zip'): 
        csv_grab_one_write(zfile,write_path,nrows = nrows)
            
def csv_grab_names_write(read_path=READ_PATH,write_path=WRITE_PATH):
    ''' This function will make a sorted list of all field names in the delrejse*.zip
    files. this facilitates fast access to the feather files when only a subset of fields are needed'''
    
    rejsekort_names = sorted({c for zfile in read_path.glob('delrejse*.zip') 
            for c in pd.read_csv(zfile,encoding='iso-8859-1',nrows  = 10 ).columns})
    with open(write_path /'delrejse_fields.txt','wt') as f:
        f.write(' '.join(rejsekort_names))
    
if __name__ == "__main__":        
    csv_grab_names_write()
    csv_grab_all_write(nrows=2_000_000_000)        
