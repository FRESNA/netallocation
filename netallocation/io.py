#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:58:09 2019

@author: fabian
"""

import sparse, os
from pathlib import Path
import xarray as xr
from progressbar import ProgressBar
import logging
from h5py import File
import numpy as np

multi_index_levels = dict(branch = ['component', 'branch_i'],
                          production = ['source', 'source_carrier'],
                          demand = ['sink', 'sink_carrier'])


def sparse_to_h5(coo, file, name):
    """
    Store sparse data in hdf5 format.

    Fast way to store sparse.COO arrays.

    Parameters
    ----------
    coo : spare.COO
        Data array which will be stored.
    file : str
        Name of the h5 file in which the data should be stored.
    name : str
        Variable name of the data under which is will be stored in the h5 file.

    Returns
    -------
    None.

    """
    hf = File(file, 'a')
    hf.create_dataset(name, data=np.vstack([coo.coords, coo.data]))
    hf.close()


def read_sparse_h5(file, name, shape):
    """
    Load the a sparse data array stored via `sparse_to_h5`.

    Parameters
    ----------
    file : str
        Name of the h5 file from where the data should be loaded.
    name : str
        Name of the variable stored in the h5 file.

    Returns
    -------
    sparse.COO
        Loaded sparse array.

    """
    hf = File(file, 'r')
    raw = np.array(hf.get(name))
    hf.close()
    coords = raw[:-1]
    data = raw[-1]
    return sparse.COO(coords, data, shape=shape)


def dense_to_h5(array, file, name):
    """
    Store dense data in hdf5 format.

    Fast way to store arrays, helpful for datasets with stacked multindex.

    Parameters
    ----------
    coo : spare.COO
        Data array which will be stored.
    file : str
        Name of the h5 file in which the data should be stored.
    name : str
        Variable name of the data under which is will be stored in the h5 file.

    Returns
    -------
    None.

    """
    hf = File(file, 'a')
    hf.create_dataset(name, data=array)
    hf.close()


def read_dense_h5(file, name, shape):
    """
    Load the a dense data array stored via `dense_to_h5`.

    Parameters
    ----------
    file : str
        Name of the h5 file from where the data should be loaded.
    name : str
        Name of the variable stored in the h5 file.

    Returns
    -------
    dense.COO
        Loaded dense array.

    """
    hf = File(file, 'r')
    array = np.array(hf.get(name))
    hf.close()
    return array



coord_fn = 'coords.nc'
sparse_fn = 'sparse_data.h5'

def store_sparse_dataset(dataset, folder):
    """
    Export xarray.Dataset with sparse Dataarrays.

    Use this to write out xarray.Datsets with sparse arrays
    and multiindex. Those will be stored as npz files in a new created
    directory.


    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with sparse variables to be written
    folder : str
        Name of the directory to where to data is stored.

    Returns
    -------
    None.

    """
    ds = dataset.copy()
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    for d in [coord_fn, sparse_fn]:
        if p.joinpath(d).exists():
            os.remove(p.joinpath(d))
    sp_vars = [v for v in ds if isinstance(ds[v].data, sparse.COO)]
    ds = ds.assign_attrs(**{'_dims_' + v: ds[v].dims for v in sp_vars})
    ds = ds.assign_attrs(**{'_shape_' + v: ds[v].shape for v in sp_vars})
    progress = ProgressBar()
    logging.info(f'Storing {len(sp_vars)} sparse datasets.')
    if len(sp_vars):
        for v in progress(sp_vars):
            sparse_to_h5(ds[v].data, p.joinpath(sparse_fn), v)
            ds = ds.drop(v)
    cp = p.joinpath(coord_fn)
    reset_multi = [k for k in multi_index_levels if k in ds.coords]
    ds.reset_index(reset_multi).to_netcdf(cp)


def load_sparse_dataset(folder):
    """
    Import xarray.Dataset with sparse Dataarrays.

    Use this to load an xarray.Dataset stored via the function
    `store_sparse_dataset`.

    Parameters
    ----------
    folder : str
        Directory name of the stored data.

    Returns
    -------
    xarray.Dataset
        Allocation dataset with sparse data arrays.

    """
    p = Path(folder)
    ds = xr.load_dataset(p.joinpath(coord_fn))
    set_index = {k: v for k,v in multi_index_levels.items() if k in ds.dims}
    ds = ds.set_index(set_index)
    vars = [v[6:] for v in ds.attrs.keys() if v.startswith('_dim')]
    for v in vars:
        dims = ds.attrs.pop('_dims_' + v)
        shape = tuple(ds.attrs.pop('_shape_' + v))
        data = read_sparse_h5(p.joinpath(sparse_fn), v, shape=shape)
        ds = ds.assign({v: (dims , data)})
    return ds
