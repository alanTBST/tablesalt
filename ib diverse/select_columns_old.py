# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
import pandas as pd
from pathlib import Path
import datetime
import gc
import fnmatch

path = Path(r'data2/') # path to feather files (.ft)
# zfile = Path(r'//tsdw03/h/Data/Rejsedata/2019/Delrejser/delrejse_2019_12_12.zip')
# zfile = Path(r'data/delrejse_2019_12_12.zip')

def get_columns_names(df,pat):
    '''Return names of columns in dataframe which we want to retrieve'''
    wanted = [n for n in df.columns if any([fnmatch.fnmatch(n, pat) for pat in pat.split(' ')])]
    return wanted 

def get_columns_from_ft(ftfile,pat='*',filter = ''):
    '''Retrieves columns from the delrejse data filter it
    '''
    if type(pat)==str:
        with open(ftfile,'br') as f:    
            df = pd.read_feather(f)
        wanted = get_columns_names(df, pat)
    else: 
        with open(ftfile,'br') as f:    
            df = pd.read_feather(f,columns=pat)
        wanted = pat
    
    if filter:
        res = df.query(filter).loc[:,wanted]
    else:
        res = df.loc[:,wanted]
    print(f'Reading {ftfile}')
    return res

def get_columns_names_from_ft(ftfile):
    '''Retrieves columns from the delrejse data filter it
    '''
    with open(ftfile,'br') as f:    
        df = pd.read_feather(f)
    col = df.columns
    print(f'Reading {ftfile}')
    return col

 #%% 
def get_dataframe(pat='*',filter= '',nmax=1000):
    '''Returns a dataframe with columns fullfil pat and rows fulfill 
    filter'''
    
    allthis = pd.concat([get_columns_from_ft(ftfile,pat,filter) 
        for number,ftfile in  enumerate(path.glob('delrejse*.ft'))
        if number < nmax])
    print(f'Memory usage for pattern:"{pat}" {allthis.memory_usage().sum():,}')
    return allthis

def delrejsecolumns(nmax=1):
    '''Find the columns names contained in the .ft files
    set nmax to a higher number to investigate if column names 
    are changing'''
    
    allnames_list  = [get_columns_names_from_ft(ftfile)
                         for number,ftfile in  enumerate(path.glob('delrejse*.ft')) if number < nmax]
    allnames = {n  for l in allnames_list  for n in l}
    return sorted(allnames)

def try_column(nmax=1):
    '''Find the columns names contained in the .ft files
    set nmax to a higher number to investigate if column names 
    are changing'''
    
    for number,ftfile in  enumerate(path.glob('delrejse*.ft')):
         if number < nmax:
            with open(ftfile,'br') as f:    
                df = pd.read_feather(f,columns=['Model'])
    return df
    
            
    
    allnames = {n  for l in allnames_list  for n in l}
    return sorted(allnames)

if __name__ == "__main__":
    if 0:
        allnames = delrejsecolumns(nmax=1)
        prisdf = get_dataframe('msg* *pris', 'RejsePris > 0 & RejsePris <100',nmax=1)
        delrejse_short  = get_dataframe('*', '',nmax=1).head(1000)[allnames]
    #t=try_column()
    prisdf = get_dataframe('RejsePris', 'RejsePris > 0 & RejsePris <100',nmax=100)
    prisdf2 = get_dataframe(['RejsePris'], 'RejsePris > 0 & RejsePris <100',nmax=100)
