#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:10:07 2019

@author: fabian
"""

import pandas as pd
import logging
logger = logging.getLogger(__name__)
import sparse
import numpy as np
import xarray as xr
from sparse import as_coo

def upper(df):
    return df.clip(min=0)

def lower(df):
    return df.clip(max=0)


def get_branches_i(n, branch_components=None):
    if branch_components is None: branch_components = n.branch_components
    return pd.concat((n.df(c)[[]] for c in branch_components),
           keys=branch_components).index.rename(['component', 'branch_i'])

def filter_null(da, dim=None):
    if dim is not None:
        return da.where(da != 0).dropna(dim, how='all')
    return da.where(da != 0)

def array_as_sparse(da):
    return da.copy(data=as_coo(da.data))

def as_sparse(ds):
    if isinstance(ds, xr.Dataset):
        return ds.assign(**{k: array_as_sparse(ds[k]) for k in ds})
    else:
        return array_as_sparse(ds)

def array_as_dense(da):
    return da.copy(data=da.data.todense())

def as_dense(ds):
    if isinstance(ds, xr.Dataset):
        return ds.assign(**{k: array_as_dense(ds[k]) for k in ds})
    else:
        return array_as_dense(ds)


def parmap(f, arg_list, nprocs=None, **kwargs):
    import multiprocessing

    def fun(f, q_in, q_out):
        while True:
            i, x = q_in.get()
            if i is None:
                break
            q_out.put((i, f(x, **kwargs)))

    if nprocs is None:
        nprocs = multiprocessing.cpu_count()
    logger.info('Run process with {} parallel threads.'.format(nprocs))
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    proc = [multiprocessing.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(arg_list)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]


