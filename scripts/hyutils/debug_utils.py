# Created by: Hunter Young
# Date: 4/24/18
#
# Script Description:
# 	TODO
# Current Recommended Usage: (in terminal)
# 	TODO

import os, sys
import numpy as np
import pandas as pd

def plist(head,values,dplace=2,sep=', '):
    try:
        if(isinstance(values[0],int)): print(head + sep.join(str(val) for val in values))
        elif(isinstance(values[0],float)): print(head + sep.join(str(round(val,dplace)) for val in values))
        else: print(head + sep.join(str(val) for val in values))
    except:
        print(head + sep.join(str(val) for val in values))
        pass

def pmat(mat,col_lbls,row_lbls, head=None):
    df = pd.DataFrame(mat, columns=col_lbls, index=row_lbls)
    if head is not None: display(HTML('<h3>'+ head +'</h3>'))
    display(df)
