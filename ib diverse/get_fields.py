# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
import pandas as pd
from pathlib import Path
import datetime
import gc
import fnmatch

path = Path(r'data/') # path to feather files (.ft)

def get_columns_from_ft(ftfile,wanted=None,filter = ''):
    '''Retrieves columns from the delrejse data filter it
    '''
    print(f'Reading {ftfile}')
    with open(ftfile,'br') as f:    
          df = pd.read_feather(f,columns=wanted)
    return df.query(filter).loc[:,wanted] if filter else df.loc[:,wanted]


def get_field_names(read_path=path):
    ''' This function will read a sorted list of all field names in the delrejse*.zip
    files. this facilitates fast access to the feather files when only a subset of fields are needed'''
    
    with open(read_path /'delrejse_fields.txt','rt') as f:
        field_names = f.read().split()
    return field_names

def get_dataframe(pat='*',filter= '',nmax=1000,nmin=0,head=0):
    '''Returns a dataframe with columns fullfil pat and rows fulfill 
    filter
    
    nmax is the max number of feather files
    nmin is the min number of feather files
    
    head is the number of lines returned, all if head == 0 '''
    
    field_names = get_field_names()
    if pat == '*' and False:
        fields_wanted=None
    else:
        fields_wanted = [n for n in field_names 
                     if any([fnmatch.fnmatch(n, pat) for pat in pat.split(' ')])]

    
    allthis = pd.concat([get_columns_from_ft(ftfile,fields_wanted,filter) 
        for number,ftfile in  enumerate(path.glob('delrejse*.ft'))
        if nmin <= number < nmax],ignore_index=True)
    outdf = allthis.head(head).copy() if head else allthis
    print(f'Memory usage for pattern:"{pat}" {outdf.memory_usage().sum():,}')
    return outdf


if __name__ == "__main__":
    sjalland =  {'Vestsjælland', 'Hovedstadsområdet','Sydsjælland'}

    field_names = get_field_names()
    delrejse_short  = get_dataframe('*', '',nmax=1,nmin=0,head=2000).query('Stop2Takst in @sjalland')
    delrejse_short.to_excel('delrejse_short.xlsx')
    delrejse_short_t = delrejse_short.T
    if 0:
        prisdf = get_dataframe('msg* *pris', 'RejsePris > 0 & RejsePris <100',nmax=3)
        delrejse_short  = get_dataframe('*', '',nmax=1,nmin=0).head(1000)
        delrejse_short_t = delrejse_short.T
        #t=try_column()
        prisdf = get_dataframe('RejsePris', 'RejsePris > 0 & RejsePris <100',nmax=33)
#%% 
    fields = 'ModalType   Msgreportdate  Stop2Takst Model StopPointNr'.split()
    fields_index = pd.MultiIndex.from_frame(delrejse_short['turngl Applicationtransactionsequencenu'.split()].rename({'Applicationtransactionsequencenu':'Seq'},axis=1))
    inspect = delrejse_short[fields].set_index(fields_index)
    inspect.query('Stop2Takst in @sjalland')
