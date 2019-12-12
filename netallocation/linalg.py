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
    return df.clip(min=0)

def lower(df):
    return df.clip(max=0)


def pinv(df):
    return DataArray(np.linalg.pinv(df), df.T.coords)

def inv(df, pre_clean=False):
    if pre_clean:
        zeros_b = df == 0
        mask = tuple(df.get_index(d)[~zeros_b.all(d).values] for d in df.dims)
        subdf = df.loc[mask]
        return DataArray(np.linalg.inv(subdf), subdf.T.coords, subdf.dims[::-1])\
                        .reindex(df.T.coords, fill_value=0)
    return DataArray(np.linalg.inv(df), df.T.coords)


def dedup_axis(da, newdims):
    """
    Helper function for DataArrays which have two dimensions using the same
    coordinates (duplicate dimensions), like (bus, bus). This sets a new
    DataArray with new names for the new coordinates.
    """
    oldindex = da.get_index(da.dims[0])
    assert not isinstance(oldindex, pd.MultiIndex), ('Multiindex expanding not '
                         'supported')
    return DataArray(da.values, {newdims[0]: oldindex.rename(newdims[0]),
                     newdims[1]: oldindex.rename(newdims[1])}, newdims)



def _dot_single(df, df2):
    dim0 = df.dims[0]
    dim1 = df2.dims[-1]
    assert df.get_index(df.dims[-1]).equals(df2.get_index(df2.dims[0]))
    res = df.values @ df2.values
    if res.ndim == 1:
        return DataArray(res, {dim0: df.coords.indexes[dim0]}, dim0)
    return DataArray(res, {dim0: df.coords.indexes[dim0],
                           dim1: df2.coords.indexes[dim1]}, (dim0, dim1))


def dot(*das):
    """
    Perform a matrix-multiplication for two or more xarray.DataArrays. This is
    different to the xarray dot-product which is a tensor-product
    """
    return reduce(_dot_single, das)

def null(df):
    if not df.size:
        return df
    dim = df.dims[-1]
    return DataArray(pd.DataFrame(sp.linalg.null_space(df), index=df.get_index(dim)),
                     dims=(dim, 'null_vectors'))


def diag(da, newdims=None):
    """
    Convenience function to select diagonal from a square matrix, or to build
    a diagonal matrix from a 1 dimensional array.

    Parameters
    ----------
    da : xarray.DataArray
    """
    if newdims is not None:
        oldindex = da.get_index(da.dims[0])
        return DataArray(np.diag(da), {newdims[0]: oldindex.rename(newdims[0]),
                     newdims[1]: oldindex.rename(newdims[1])}, newdims)
    if da.ndim == 1:
        return DataArray(np.diag(da), dims=da.dims * 2, coords=da.coords)
    return DataArray(np.diagflat(np.diag(da)), da.coords)



def eig(M):
    val, vec = np.linalg.eig(M)
    val = pd.Series(val).sort_values(ascending=False)
    vec = pd.DataFrame(vec, index=M.index).reindex(columns=val.index)
    return val, vec
