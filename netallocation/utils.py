#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains tool functions for making the life easier in the other
modules.
"""

import pandas as pd
import xarray as xr
from pypsa.geo import haversine_pts
from pypsa.descriptors import (get_extendable_i, get_non_extendable_i,
                               nominal_attrs, get_switchable_as_dense)
from sparse import as_coo, COO

def upper(ds):
    "Clip all negative entries of a xr.Dataset/xr.DataArray."
    ds = obj_if_acc(ds)
    return ds.clip(min=0)

def lower(ds):
    "Clip all positive entries of a xr.Dataset/xr.DataArray."
    ds = obj_if_acc(ds)
    return ds.clip(max=0)

def get_branches_i(n, branch_components=None):
    "Get a pd.Multiindex for all branches in the network."
    branch_components = check_branch_comps(branch_components, n)
    return pd.concat({c: n.df(c)[[]] for c in branch_components})\
                    .index.rename(['component', 'branch_i'])

def get_ext_branches_i(n, branch_components=None):
    "Get a pd.Multiindex for all extendable branches in the network."
    branch_components = check_branch_comps(branch_components, n)
    return pd.Index(sum(([(c, i) for i in get_extendable_i(n, c)]
                                 for c in branch_components), []))

def get_non_ext_branches_i(n, branch_components=None):
    "Get a pd.Multiindex for all non-extendable branches in the network."
    branch_components = check_branch_comps(branch_components, n)
    return pd.Index(sum(([(c, i) for i in get_non_extendable_i(n, c)]
                                 for c in branch_components), []))

def get_ext_one_ports_i(n, per_carrier=True):
    "Get a pd.Multiindex for all extendable branches in the network."
    check_carriers(n)
    comps = n.one_port_components & set(nominal_attrs)
    if per_carrier:
        return pd.MultiIndex.from_frame(pd.concat(n.df(c).loc[
                get_extendable_i(n, c), ['bus', 'carrier']] for c in comps))
    return pd.Index(sum(([(c, i) for i in get_extendable_i(n, c)]
                                 for c in comps), []))

def get_non_ext_one_ports_i(n, per_carrier=True):
    "Get a pd.Multiindex for all non-extendable branches in the network."
    check_carriers(n)
    comps = n.one_port_components & set(nominal_attrs)
    if per_carrier:
        return pd.MultiIndex.from_frame(pd.concat(n.df(c).loc[
                get_non_extendable_i(n, c), ['bus', 'carrier']] for c in comps))
    return pd.Index(sum(([(c, i) for i in get_non_extendable_i(n, c)]
                                 for c in comps), []))

def get_ext_one_ports_b(n):
    check_carriers(n)
    gen = [reindex_by_bus_carrier(n.df(c)[attr + '_extendable'], c, n)
           for c,attr in nominal_attrs.items() if c in n.one_port_components]
    return xr.concat(gen, dim='carrier').astype(bool)

def get_ext_branches_b(n):
    ds = pd.concat({c: n.df(c)[attr + '_extendable'] for c, attr
                   in nominal_attrs.items() if c in sorted(n.branch_components)},
                   names=['component', 'branch_i'])
    return xr.DataArray(ds, dims='branch')


def split_one_ports(ds, n, dim='bus'):
    "Split data into extendable one ports and nonextendable one ports"
    assert 'carrier' in ds.dims, "Dimension 'carrier' not in dataset."
    ext_b = get_ext_one_ports_b(n).rename(bus=dim)
    d = {'ext': ds.where(ext_b).fillna(0),
         'fix': ds.where(~ext_b).fillna(0)}
    return xr.Dataset(d) if isinstance(ds, xr.DataArray) else d


def split_branches(ds, n):
    "Split data into extendable one ports and nonextendable one ports"
    ext_b = get_ext_branches_b(n)
    d = {'ext': ds.where(ext_b).fillna(0),
         'fix': ds.where(~ext_b).fillna(0)}
    return xr.Dataset(d) if isinstance(ds, xr.DataArray) else d


def generation_carriers(n):
    return n.generators.carrier.unique()

def snapshot_weightings(n, snapshots=None):
    snapshots = check_snapshots(snapshots, n)
    w = n.snapshot_weightings.loc[snapshots]
    if isinstance(w, pd.Series):
        return xr.DataArray(w, dims='snapshot')
    else:
        return w

def filter_null(da, dim=None):
    "Drop all coordinates with only null/nan entries on dimensions dim."
    da = obj_if_acc(da)
    if dim is not None:
        return da.where(da != 0).dropna(dim, how='all')
    return da.where(da != 0)

def as_sparse(ds):
    """
    Convert dense dataset/dataarray into a sparse dataset.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray

    Returns
    -------
    xr.Dataset or xr.DataArray
        Dataset or DataArray with sparse data.

    """
    ds = obj_if_acc(ds)
    func = lambda data: COO(data) if not isinstance(data, COO) else data
    return xr.apply_ufunc(func, ds)


def as_dense(ds):
    """
    Convert sparse dataset/dataarray into a dense dataset.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray

    Returns
    -------
    xr.Dataset or xr.DataArray
        Dataset or DataArray with dense data.

    """
    ds = obj_if_acc(ds)
    func = lambda data: data.todense() if isinstance(data, COO) else data
    return xr.apply_ufunc(func, ds)

def is_sparse(ds):
    """
    Check if a xarray.Dataset or a xarray.DataArray is sparse.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray

    Returns
    -------
    Bool

    """
    if isinstance(ds, xr.Dataset):
        return all(isinstance(ds[v].data, COO) for v in ds)
    else:
        return isinstance(ds.data, COO)


def obj_if_acc(obj):
    """
    Get the object of underying Accessor if Accessor is passed.

    This function is usefull to straightforwardly make functions through the
    AllocationAccessor accessible.

    Parameters
    ----------
    obj : AllocationAccessor or xarray.Dataset

    Returns
    -------
    obj
        Dataset of the accessor if accessor was is passed, ingoing object
        otherwise.

    """
    from .common import AllocationAccessor
    if isinstance(obj, AllocationAccessor):
        return obj._obj
    else:
        return obj

def bus_distances(n):
    """
    Calculate the geographical distances between all buses.

    Parameters
    ----------
    n : pypsa.Network

    Returns
    -------
    xr.DataArray
        Distance matrix of size N x N, with N being the number of buses.

    """
    xy = n.buses[['x', 'y']]
    d = xy.apply(lambda ds: pd.Series(haversine_pts(ds, xy), xy.index), axis=1)
    return xr.DataArray(d, dims=['source', 'sink'])


def reindex_by_bus_carrier(df, c, n):
    """
    Group a time-dependent dataframe by bus and carrier.

    Parameters
    ----------
    df : pd.DataFrame
        Time-dependent series to group, e.g. n.generators_t.p
    c : str
        Component name of the underlying data.
    n : pypsa.Network

    Returns
    -------
    xarray.DataArray
        Grouped array with dimension ('snapshot', 'bus', 'carrier').

    """
    check_duplicated_carrier(n)
    if isinstance(df, pd.DataFrame):
        df = df.rename(columns=n.df(c)[['bus', 'carrier']].apply(tuple, axis=1))
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['bus', 'carrier'])
        return xr.DataArray(df, dims=['snapshot', 'dim_1'])\
                 .unstack('dim_1', fill_value=0)
    else:
        df = df.rename(n.df(c)[['bus', 'carrier']].apply(tuple, axis=1))
        df.index = pd.MultiIndex.from_tuples(df.index, names=['bus', 'carrier'])
        return xr.DataArray(df).unstack('dim_0', fill_value=0)


def get_as_dense_by_bus_carrier(n, attr, comps=None, snapshots=None):
    snapshots = check_snapshots(snapshots, n)
    comps = check_one_port_comps(comps, n)
    buses_i = n.buses.index
    return xr.concat(
        (reindex_by_bus_carrier(
            get_switchable_as_dense(n, c, attr, snapshots), c, n)
         for c in comps), dim='carrier').reindex(bus=buses_i, fill_value=0)


def check_dataset(ds):
    """
    Ensure the argument is an xarray.Dataset.

    If the argument was a Dataset, the a tupel (ds, True) is returned, if it
    isn't' a Dataset a tuple of the (Dataset(ds), False) is returned.
    """
    if isinstance(ds, xr.Dataset):
        return ds, True
    else:
        name = 'variable' if ds.name is None else ds.name
        return xr.Dataset({name: ds}), False


def check_duplicated_carrier(n):
    check_carriers(n)
    dupl = pd.Series({c: n.df(c).carrier.unique() for c in n.one_port_components
                      if 'carrier' in n.df(c)}).explode()\
                      [lambda ds: ds.duplicated(keep=False)].dropna()
    assert dupl.empty, (f'The carrier name(s) {dupl.to_list()} appear in more '
                        f'than one component {dupl.index}. This will not work '
                        'when spanning the bus x carrier dimensions. Please '
                        'ensure unique carrier names.')


def check_carriers(n):
    """
    Ensure if carrier of stores is defined.

    Parameters
    ----------
    n : pypsa.Network

    """
    if 'carrier' not in n.loads:
        n.loads['carrier'] = 'Load'
    if 'carrier' not in n.stores:
        n.stores['carrier'] = n.stores.bus.map(n.buses.carrier)
    if 'carrier' not in n.lines:
        n.lines['carrier'] = n.lines.bus0.map(n.buses.carrier)
    if 'carrier' not in n.links:
        n.links['carrier'] = n.links.bus0.map(n.buses.carrier)

def check_snapshots(arg, n):
    """
    Set argument to n.snapshots if None
    """
    if isinstance(arg, pd.Index):
        return arg.rename('snapshot')
    if isinstance(arg, xr.DataArray):
        if not arg.dims:
            return arg
        return arg.to_index()
    return n.snapshots.rename('snapshot') if arg is None else arg

def set_default_if_none(arg, n, attr):
    """
    Set any argument to an attribute of n if None
    """
    return sorted(getattr(n, attr)) if arg is None else sorted(arg)

def check_passive_branch_comps(arg, n):
    """
    Set argument to n.passive_branch_components if None
    """
    return set_default_if_none(arg, n, 'passive_branch_components')

def check_branch_comps(arg, n):
    """
    Set argument to n.branch_components if None
    """
    return set_default_if_none(arg, n, 'branch_components')

def check_one_port_comps(arg, n):
    """
    Set argument to n.one_port_components if None
    """
    return set_default_if_none(arg, n, 'one_port_components')

