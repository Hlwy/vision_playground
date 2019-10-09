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

def pmat(mat,col_lbls,row_lbls, head=None):
    df = pd.DataFrame(mat, columns=col_lbls, index=row_lbls)
    if head is not None: display(HTML('<h3>'+ head +'</h3>'))
    display(df)
