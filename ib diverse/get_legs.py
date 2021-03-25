# -*- coding: utf-8 -*-
"""
Spyder Editor


"""
import pandas as pd
# import modin.pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json 
import fnmatch
from pathlib import Path


import get_fields as gf

   


def get_a_leg(ftfile):
    '''Select all legs from Sjælland in the file ftfile which contains a chunk of delrejse data lines. 
    
    A leg consists of  [StopPointNr_legstart StoppointNr ModalType]  from Delrejse 
    where:
        
      - Model is in leg_end (set defined below) and Model_legstart is in leg_start (set defined below). 
    
    
     - StopPointNr_legstart is StopPointNr from previous record. 
     - Model_legstart is Model from previous record 
     - Modal_type is the type of transport in ['2', '4', '1'] 
        - 1 - Metro 
        - 2 - Bus
        - 4 - Tog 
          
    records are filteret by the Stop2Takst in the variable Sjalland defined below.  
    
    '''
    
    leg_end = {'Co','Su'}
    leg_start = {'Fi','Tr','Su'}
    sjalland =  {'Vestsjælland', 'Hovedstadsområdet','Sydsjælland'}
    tdict = {'1':'Metro','2':'bus','4':'train'}
    
    
    with open(ftfile,'br') as f:    
         legdf = pd.read_feather(f,columns='ModalType Msgreportdate  Stop2Takst Model StopPointNr'.split())
     
    
    legdf.loc[:,'StopPointNr_leg_start'] = legdf.loc[:,'StopPointNr'].shift()  
    legdf.loc[:,'Model_leg_start']       = legdf.loc[:,'Model'].shift() 
    legdf.loc[:,'Stop2Takst_start']       = legdf.loc[:,'Stop2Takst'].shift() 


    leg_out = legdf\
         .query('Stop2Takst in @sjalland and Stop2Takst_start in @sjalland and '+
          'Model in @leg_end and Model_leg_start in @leg_start'+\
           ' and StopPointNr_leg_start != "0"' )\
            ['ModalType StopPointNr_leg_start StopPointNr '.split()].copy().rename(
             columns = {'StopPointNr':'StopPointNr_leg_end'}).replace({'ModalType':tdict}).astype('category')

    return leg_out 
             
def get_all_legs(nmin=0,nmax=200):
    path = Path(r'data/') # path to feather files (.ft)

    alllegs = pd.concat([get_a_leg(ftfile) 
     for number,ftfile in  enumerate(path.glob('delrejse*.ft'))
     if nmin <= number < nmax],ignore_index=True).astype('category')
    
    print(f'{len(alllegs):,} Rejsekort legs are found. Memory = {alllegs.memory_usage().sum()/1048576 :,.0f} MB')
    

    return alllegs   



if __name__ == "__main__":
    legs = get_all_legs()
    unique_all_modal  = set(legs.itertuples(index=False,name=None))
    unique = set(legs[['StopPointNr_leg_start','StopPointNr_leg_end']].itertuples(index=False,name=None))
    uniquedf = pd.DataFrame(unique,columns=['StopPointNr_leg_start','StopPointNr_leg_end'],dtype='category')
    unique_all_modaldf =  pd.DataFrame(unique_all_modal,columns=legs.columns,dtype='category')
    print(f'{len(uniquedf):19,} unique trips')
    #%% calculations 
    legs_modal_count = unique_all_modaldf.ModalType.value_counts() 
    
    legs_start_count = uniquedf.StopPointNr_leg_start.value_counts()
    legs_end_count   = uniquedf.StopPointNr_leg_end.value_counts()
    potential_count = len(legs_start_count)*len(legs_end_count)
    print(f'{potential_count:19,} potential trips')
    
    StopPointNr_from_legs = {int(i) for i in pd.concat([uniquedf.StopPointNr_leg_start,uniquedf.StopPointNr_leg_end]).unique()}
    try: 
        points_only_in_legs =  StopPointNr_from_legs -StopPointNr_from_zone_map
        points_only_in__zone_map  =  StopPointNr_from_zone_map - StopPointNr_from_legs
    except:
        print('not able to run comparaison to zonemap, first run test_topo and explore_stoppoints')
        pass
    #%%
    xx = sorted(points_only_in_legs)
    out=[]
    for i,point in enumerate(sorted(points_only_in_legs)):
        if i >=100000: break 
        this = f'{i:5} Point in legs not in zone_map: {point:6}'+\
             ' '.join(f' {id} ' for id,nr in unique_stoppoints if int(nr) == point)
        out.append(this)     
    
    