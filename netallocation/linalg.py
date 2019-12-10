#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:17:46 2019

@author: fabian
"""

import pandas as pd
from xarray import DataArray
import numpy as np
import scipy as sp
from functools import reduce


def upper(df):
    return df.clip(lower=0)

def lower(df):
    return df.clip(upper=0)


def pinv(df):
    return DataArray(np.linalg.pinv(df), df.T.coords)

def inv(df, pre_clean=False):
    if pre_clean:
        zeros_b = df == 0
        zero_rows_b = zeros_b.all(1)
        zero_cols_b = zeros_b.all()
        subdf = df[~zero_rows_b, ~zero_cols_b]
        return DataArray(np.linalg.inv(subdf), subdf.T.coords)\
                        .reindex_like(df.T).fillna(0)
    return DataArray(np.linalg.inv(df), df.T.coords)

def dot(df, df2):
    return df.dot(df2, df.dims[-1])

def dots(*das):
    """
    Chaining matrix multiplication
    """
    return reduce(dot, das)


def mdot(df, df2):
    dim0 = df.dims[0]
    dim1 = df2.dims[-1]
    assert df.get_index(df.dims[-1]).equals(df2.get_index(df2.dims[0]))
    res = df.values @ df2.values
    if res.ndim == 1:
        return DataArray(res, {dim0: df.coords.indexes[dim0]}, dim0)
    return DataArray(res, {dim0: df.coords.indexes[dim0],
                           dim1: df2.coords.indexes[dim1]}, (dim0, dim1))


def mdots(*das):
    """
    Chaining matrix multiplication
    """
    return reduce(mdot, das)

def null(df):
    if not df.size:
        return df
    dim = df.dims[-1]
    return DataArray(pd.DataFrame(sp.linalg.null_space(df), index=df.get_index(dim)),
                     dims=(dim, 'null_vectors'))


def diag(da):
    """
    Convenience function to select diagonal from a square matrix, or to build
    a diagonal matrix from a 1 dimensional array.

    Parameters
    ----------
    da : xarray.DataArray
    """
    if da.ndim == 1:
        return DataArray(np.diag(da), dims=da.dims * 2, coords=da.coords)
    return DataArray(np.diagflat(np.diag(da)), da.coords)



def eig(M):
    val, vec = np.linalg.eig(M)
    val = pd.Series(val).sort_values(ascending=False)
    vec = pd.DataFrame(vec, index=M.index).reindex(columns=val.index)
    return val, vec
