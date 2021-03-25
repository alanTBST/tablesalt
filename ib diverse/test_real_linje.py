# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from pathlib import Path
import zipfile

READ_PATH =Path(r'//tsdw03/h/Data/Rejsedata/2019/Delrejser/')


for i,zfile in enumerate(READ_PATH.glob('delrejse*.zip')): 
    if i >= 1: 
        break
    print(i,zfile)
    with zipfile.ZipFile(zfile,'r').open('delrejse_2019_12_01.csv',encoding='iso-8859-1') as f:
        print('read')
        out = f.readlines(10)
