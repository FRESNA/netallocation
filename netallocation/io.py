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


def sparse_array_to_h5(coo, file, name):
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
    hf = File(file, 'w')
    hf.create_dataset(name, data=np.vstack([coo.coords, coo.data]))
    hf.close()


def sparse_array_read_h5(file, name):
    """
    Load the a sparse data array stored via `sparse_array_to_h5`.

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
    coords = raw[:-1]
    data = raw[-1]
    return sparse.COO(coords, data, shape=tuple(coords[:, -1].astype(int)))



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
    spdas = [da for da in ds if isinstance(ds[da].data, sparse.COO)]
    ds = ds.assign_attrs(**{'_dims_' + da: ds[da].dims for da in spdas})
    progress = ProgressBar()
    logging.info(f'Storing {len(spdas)} sparse datasets.')
    for da in progress(spdas):
        sparse_array_to_h5(p.joinpath(da), ds[da].data)
        ds = ds.drop(da)
    cp = p.joinpath('coords.nc')
    reset_multi = [k for k in multi_index_levels if k in ds.coords]
    ds.drop(list(ds)).reset_index(reset_multi).to_netcdf(cp)


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
    ds = xr.load_dataset(p.joinpath('coords.nc'))
    set_index = {k: v for k,v in multi_index_levels.items() if k in ds.dims}
    ds = ds.set_index(set_index)
    name = lambda path: str(path).split(os.path.sep)[-1][:-4]
    data = {name(da): (ds.attrs.pop('_dims_' + name(da)), sparse.load_npz(da))
            for da in p.glob('*.npz')}
    return ds.assign(data)
