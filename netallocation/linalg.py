#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:17:46 2019

@author: fabian
"""

from .utils import as_sparse
import pandas as pd
from xarray import DataArray
import numpy as np
import scipy as sp
from functools import reduce
from sparse import COO, as_coo
from scipy.sparse.linalg import inv as sp_inv

def upper(df):
    return df.clip(min=0)

def lower(df):
    return df.clip(max=0)

def pinv(df):
    return DataArray(np.linalg.pinv(df), df.T.coords)

def inv(df, pre_clean=False):
    if isinstance(df.data, COO):
        if pre_clean:
            data = df.data
            assert df.ndim <= 2, ('Maximally two dimension supported for '
                                  'sparse inverse')
            mask = np.isin(np.arange(data.shape[0]), data.coords[0]) & \
                   np.isin(np.arange(data.shape[1]), data.coords[1])
            subdf = df[mask][:, mask]
            return DataArray(as_coo(sp_inv(subdf.data.tocsc())), subdf.T.coords)\
                         .reindex(**{df.dims[0]: df.get_index(df.dims[0])}, fill_value=0)\
                         .reindex(**{df.dims[1]: df.get_index(df.dims[1])}, fill_value=0)
        return DataArray(as_coo(sp_inv(df.data.tocsc())), df.T.coords)
    if pre_clean:
        zeros_b = df == 0
        mask = tuple(df.get_index(d)[~zeros_b.all(d).data] for d in df.dims)
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
    return DataArray(da.data, {newdims[0]: oldindex.rename(newdims[0]),
                     newdims[1]: oldindex.rename(newdims[1])}, newdims)



def _dot_single(df, df2):
    dim0 = df.dims[0]
    dim1 = df2.dims[-1]
    assert df.get_index(df.dims[-1]).equals(df2.get_index(df2.dims[0]))
    res = df.data @ df2.data
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


def norm(ds, dims):
    dims = [dims] if not isinstance(dims, list) else dims
    for d in dims:
        assert d in ds.dims, f'Dimension {d} not in Dataset/DataArray'
    return ds / ds.sum([d for d in ds.dims if d not in dims])


def diag(da, newdims=None, sparse=False):
    """
    Convenience function to select diagonal from a square matrix, or to build
    a diagonal matrix from a 1 dimensional array.

    Parameters
    ----------
    da : xarray.DataArray
    """
    if newdims is not None:
        oldindex = da.get_index(da.dims[0])
        res = DataArray(np.diag(da), {newdims[0]: oldindex.rename(newdims[0]),
                     newdims[1]: oldindex.rename(newdims[1])}, newdims)
    elif da.ndim == 1:
        res = DataArray(np.diag(da), dims=da.dims * 2, coords=da.coords)
    else:
        res = DataArray(np.diagflat(np.diag(da)), da.coords)
    return as_sparse(res) if sparse else res



def eig(M):
    val, vec = np.linalg.eig(M)
    val = pd.Series(val).sort_values(ascending=False)
    vec = pd.DataFrame(vec, index=M.index).reindex(columns=val.index)
    return val, vec
