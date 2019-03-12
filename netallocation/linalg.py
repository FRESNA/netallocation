#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:17:46 2019

@author: fabian
"""

import pandas as pd
import numpy as np
import scipy as sp

def pinv(df):
    return pd.DataFrame(np.linalg.pinv(df), df.columns, df.index)

def inv(df):
    return pd.DataFrame(np.linalg.inv(df), df.columns, df.index)


def null(df):
    if df.empty:
        return df
    return pd.DataFrame(sp.linalg.null_space(df), index=df.columns)


def diag(df):
    """
    Convenience function to select diagonal from a square matrix, or to build
    a diagonal matrix from a series.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
    """
    if isinstance(df, pd.DataFrame):
        if len(df.columns) == len(df.index) > 1:
            return pd.DataFrame(np.diagflat(np.diag(df)), df.index, df.columns)
    return pd.DataFrame(np.diagflat(df.values),
                        index=df.index, columns=df.index)



def eig(M):
    val, vec = np.linalg.eig(M)
    val = pd.Series(val).sort_values(ascending=False)
    vec = pd.DataFrame(vec, index=M.index).reindex(columns=val.index)
    return val, vec
