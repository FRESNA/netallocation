#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:18:23 2019

@author: fabian
"""

import pandas as pd
import xarray as xr
from xarray import DataArray
import numpy as np
from .linalg import pinv, diag, null, upper, lower, mdot, mdots, dot, dots
from .utils import get_branches_i
import logging
import networkx as nx
import scipy
from collections import Sequence
logger = logging.getLogger(__name__)



def Incidence(n, branch_components=None):
    return DataArray(n.incidence_matrix(branch_components).todense(),
              coords=(n.buses.index, get_branches_i(n, branch_components)),
              dims=['bus', 'branch']).sortby('component')

def Cycles(n, branches_i=None):
    """
    Light-weight function for finding all cycles a given network.

    """
    branches = pd.concat({c: n.df(c)[['bus0', 'bus1']] for c in
                          sorted(n.branch_components)})\
                        .rename(columns={'bus0': 'source', 'bus1': 'target'})
    if branches_i is None:
        branches_i = branches.index.rename(['component', 'branch_i'])
    else:
        branches = branches.loc[branches_i]
    branches = branches.assign(index = branches_i)
    branches_bus0 = branches['source']
    mgraph = nx.from_pandas_edgelist(branches, edge_attr=True)
    graph = nx.OrderedGraph(mgraph)
    cycles = nx.cycle_basis(graph)
    #number of 2-edge cycles
    num_multi = len(mgraph.edges()) - len(graph.edges())
    C = scipy.sparse.dok_matrix((len(branches_bus0), len(cycles) + num_multi))
    for j,cycle in enumerate(cycles):
        for i, start in enumerate(cycle):
            end = cycle[(i+1)%len(cycle)]
            branch = mgraph[start][end]['index']
            branch_i = branches.index.get_loc(branch)
            sign = +1 if branches_bus0.iat[branch_i] == cycle[i] else -1
            C[branch_i,j] += sign
    #counter for multis
    c = len(cycles)
    #add multi-graph 2-edge cycles for multiple branches between same pairs of buses
    for u,v in graph.edges():
        bs = list(mgraph[u][v].keys())
        if len(bs) > 1:
            first = bs[0]
            first_i = branches_i.get_loc(first)
            for b in bs[1:]:
                b_i = branches_i.get_loc(b)
                sign = -1 if branches_bus0.iat[b_i] == branches_bus0.iat[first_i] else +1
                C[first_i,c] = 1
                C[b_i,c] = sign
                c+=1
    return DataArray(C.todense(),  {'branch': branches_i, 'cycle': range(C.shape[1])},
                    ('branch', 'cycle'))


def impedance(n, branch_components=None, snapshot=None,
              pu_system=True, linear=True, skip_pre=True):
    #standard impedance, note z must not be inf or nan
    x = 'x_pu' if pu_system else 'x'
    r = 'r_pu' if pu_system else 'r'

    if not skip_pre:
        if pu_system and (n.lines[x] == 0).all():
            n.calculate_dependent_values()

    if branch_components is None:
        branch_components = n.passive_branch_components
    comps = sorted(set(branch_components) & n.passive_branch_components)
    if linear:
        z = pd.concat([n.df(c)[x].where(n.df(c).bus0.map(n.buses.carrier) == 'AC',
                    n.df(c)[r]) for c in comps], keys=comps)
    else:
        z = pd.concat([n.df(c).eval(f'{r} + 1.j * {x}') for c in comps], keys=comps)
    if not n.lines.empty:
        assert not np.isinf(z).any() | z.isna().any(), ('There '
        f'seems to be a problem with your {x} or {r} values. At least one of '
        f'these is nan or inf. Please check the values in components {comps}.')
    z = DataArray(z.rename_axis(['component', 'branch_i']), dims='branch')

    if ('Link' not in branch_components) | n.links.empty :
        return z

    # add pseudo impedance for links, in dependence on the current flow:
    if snapshot is None:
        logger.warn('Link in argument "branch_components", but no '
                        'snapshot given. Falling back to first snapshot')
        snapshot = n.snapshots[0]

    f = network_flow(n, snapshot)
    f_sub = f[abs(f) > 1e-8] # only branches with nonzero flow
    z_sub = z.reindex_like(f_sub)
    C = Cycles(n, f_sub.get_index('branch'))
    # C_mix is all the active cycles where at least one link is included
    C_mix = C[:, (C != 0).groupby('component').any().loc['Link'].values]

    if not C_mix.size:
        omega = DataArray(1, f_sub.loc['Link'].coords)
    elif not z.size:
        omega = null(C_mix.loc['Link'] * f_sub.loc['Link'])[0]
    else:
        d = {'branch': 'Link'}
        omega = - mdot(pinv(mdot(C_mix.loc['Link'].T, diag(f_sub.loc['Link']))),
           mdots(C_mix.drop_sel(d).T, diag(z_sub.drop_sel(d)), f_sub.drop_sel(d)))

    omega = omega.round(15).assign_coords({'component':'Link'})
    omega[(omega == 0) & (f_sub.loc['Link'] != 0)] = 1
    Z = z.reindex_like(f).copy()
    Z.loc['Link'] = omega.reindex_like(f.loc['Link'], fill_value=0)
    return Z

def admittance(n, branch_components=None, snapshot=None,
               pu_system=True, linear=True):
    y = 1/impedance(n, branch_components, snapshot, pu_system, linear)
    return y.where(~np.isinf(y), 0)


def series_shunt_admittance(n, pu_system=True, branch_components=None):
    if branch_components is None:
        branch_components = n.passive_branch_components
    g, b = ('g_pu', 'b_pu') if pu_system else ('g', 'b')
    shunt = pd.concat({c.name: c.df[g].fillna(0) + 1.j * c.df[b].fillna(0)
                       for c in n.iterate_components(branch_components)})
    return DataArray(shunt.rename_axis(['component', 'branch_i']), dims='branch')


def shunt_admittance(n, pu_system=True, branch_components=None):
    if branch_components is None:
        branch_components = n.passive_branch_components
    g, b = ('g_pu', 'b_pu') if pu_system else ('g', 'b')
    K = Incidence(n, branch_components=branch_components)
    series_shunt = series_shunt_admittance(n, pu_system)
    nodal_shunt = n.shunt_impedances.fillna(0).groupby('bus').sum()\
                    .eval(f'{g} + 1.j * {b}')\
                    .reindex(n.buses.index, fill_value=0)
    nodal_shunt = DataArray(nodal_shunt, dims='bus')
    return 0.5 * abs(K) @ series_shunt + nodal_shunt


def PTDF(n, branch_components=None, snapshot=None, pu_system=True, update=True):
    if branch_components is None:
        branch_components = n.branch_components
    if 'Link' in branch_components or update or '_ptdf' not in n.__dir__():
        n.calculate_dependent_values()
        K = Incidence(n, branch_components)
        Y = diag(admittance(n, branch_components, snapshot, pu_system=pu_system))
        n._ptdf = mdots(Y, K.T, pinv(mdots(K, Y, K.T)))
    return n._ptdf



def CISF(n, branch_components=['Line', 'Transformer'], pu_system=True):
    n.calculate_dependent_values(), n.determine_network_topology()
    K = Incidence(n, branch_components)
    y = admittance(n, branch_components, pu_system=pu_system, linear=False)
    Z = Zbus(n, branch_components, pu_system=pu_system, linear=False)
    return diag(y) @ K.T @ Z


def Ybus(n, branch_components=None, snapshot=None, pu_system=True, linear=True):
    if branch_components is None:
        branch_components = n.passive_branch_components
    """
    Calculates the Ybub matrix (or weighited Laplacian) for a given network for
    given branch_components. If branch_component or snapshots is None, which is
    the default, they are set to n.branch_components and n.snapshots
    respectively. If 'Link' is included in branch_components, then their
    weightings derived from their current pseudo-impedance dependent on the
    current flow (see :func:`impedance`).
    """
    if branch_components is None: branch_components = n.passive_branch_components
    K = Incidence(n, branch_components)
    y = admittance(n, branch_components, snapshot, pu_system=pu_system,
                   linear=linear)
    Y = mdots(K, diag(y), K.T)
    if linear:
        return Y
    else:
        if not pu_system:
            raise NotImplementedError('Non per unit system '
                                      'for non-linear Ybus matrix not '
                                      'implemented')
        return Y + diag(shunt_admittance(n, pu_system))

        def get_Ybus(sub):
            sub.calculate_PTDF()
            sub.calculate_Y()
            return pd.DataFrame(sub.Y.todense(), index=sub.buses_o,
                             columns=sub.buses_o)
        return n.sub_networks.obj.apply(get_Ybus)



def Zbus(n, branch_components=['Line'], snapshot=None,
         pu_system=True, linear=True):
    """
    If branch_component or snapshots is None, which is
    the default, they are set to n.branch_components and n.snapshots
    respectively.
    """
    if linear:
        return pinv(Ybus(n, branch_components=branch_components,
                         snapshot=snapshot,
                         pu_system=pu_system, linear=linear))
    else:
        return Ybus(n, branch_components, snapshot,
                     pu_system=pu_system, linear=False).apply(pinv)


def voltage(n, snapshots=None, linear=True, pu_system=True):
    if linear:
        v = n.buses_t.v_ang.loc[snapshots] + 1
#        v = np.exp(- 1.j * n.buses_t.v_ang).T[snapshots]
    else:
        v = (n.buses_t.v_mag_pu * np.exp(- 1.j * n.buses_t.v_ang)).T[snapshots]

    if not pu_system:
        v *= n.buses.v_nom

    return v



def network_flow(n, snapshots=None, branch_components=None, ingoing=True,
                 linear=True):
    """
    Returns the flow of the network in an xarray.DataArray for given snapshots
    and branch_components. If branch_component or snapshots is None, which is
    the default, they are set to n.branch_components and n.snapshots
    respectively.
    """
    if branch_components is None:
        branch_components = n.branch_components
    comps = sorted(branch_components)
    snapshots = n.snapshots if snapshots is None else snapshots
    p = 'p0' if ingoing else 'p1'
    axis = int(isinstance(snapshots, (list, pd.Index)))
    f = pd.concat([n.pnl(b)[p].loc[snapshots, n.df(b).index] for b in comps],
             keys=comps, axis=axis).rename_axis(['component', 'branch_i'], axis=axis)

    f = DataArray(f, dims=['snapshot', 'branch']) if axis else DataArray(f, dims='branch')
    if linear:
        return f
    else:
        q = 'q0' if ingoing else 'q1'
        return f + 1.j * pd.concat((n.pnl(b)[q].loc[snapshots, n.df(b).index]
                                    for b in branch_components), axis=1).values


def branch_inflow(n, snapshots=None, branch_components=None):
    """
    Returns the flow that goes into a branch. If branch_component or
    snapshots is None, which is the default, they are set to n.branch_components
    and n.snapshots respectively.
    """
    f0 = network_flow(n, snapshots, branch_components).T
    f1 = network_flow(n, snapshots, branch_components, ingoing=False).T
    return f0.where(f0 > 0, - f1)


def branch_outflow(n, snapshots=None, branch_components=None):
    """
    Returns the flow that comes out of a branch. If branch_component or
    snapshots is None, which is the default, they are set to n.branch_components
    and n.snapshots respectively.
    """
    f0 = network_flow(n, snapshots, branch_components).T
    f1 = network_flow(n, snapshots, branch_components, ingoing=False).T
    return f0.where(f0 < 0,  - f1)


def network_injection(n, snapshots=None, branch_components=None):
    """
    Function to determine the total network injection including passive and
    active branches.
    """
    f0 = network_flow(n, snapshots, branch_components).T
    f1 = network_flow(n, snapshots, branch_components, ingoing=False).T
    K = Incidence(n, branch_components)
    return (K.clip(min=0) @ f0 - K.clip(max=0) @ f1).T


def power_production(n, snapshots=None,
                     components=['Generator', 'StorageUnit'],
                     per_carrier=False, update=False):
    if snapshots is None:
        snapshots = n.snapshots
    if not per_carrier:
        if 'p_plus' not in n.buses_t or update:
            n.buses_t.p_plus = DataArray(pd.concat(
                    [n.pnl(c).p.clip(0).rename(columns=n.df(c).bus)
                     for c in components], axis=1).sum(level=0, axis=1)\
                    .reindex(columns=n.buses.index, fill_value=0),
                    dims=['snapshot', 'bus'])
        return n.buses_t.p_plus.reindex({'snapshot': snapshots})
    else:
        if 'p_plus_per_carrier' not in n.buses_t or update:
            p = pd.concat(((n.pnl(c).p.T.assign(carrier=n.df(c).carrier,
                                                bus=n.df(c).bus)
                            .groupby(['bus', 'carrier']).sum().T
                            .where(lambda x: x > 0))
                            for c in components), axis=1)\
                    .rename_axis(columns=['bus', 'carrier'])
            n.buses_t.p_plus_per_carrier = DataArray(p, dims=['snapshot','p'])\
                                                    .unstack('p')
    return n.buses_t.p_plus_per_carrier.reindex({'snapshot': snapshots})


def power_demand(n, snapshots=None,
                 components=['Load', 'StorageUnit'],
                 per_carrier=False, update=False):
    if snapshots is None:
        snapshots = n.snapshots.rename('snapshot')
    if not per_carrier:
        if 'p_minus' not in n.buses_t or update:
            n.buses_t.p_minus = DataArray(
                    sum(n.pnl(c).p.T.mul(n.df(c).sign, axis=0).clip(upper=0)
                        .assign(bus=n.df(c).bus).groupby('bus').sum()
                        .reindex(index=n.buses.index, fill_value=0).T
                        for c in components).abs().rename_axis('sink', axis=1),
                    dims=['snapshot', 'bus'])
        return n.buses_t.p_minus.reindex({'snapshot': snapshots})

    if 'p_minus_per_carrier' not in n.buses_t or update:
        if 'carrier' not in n.loads:
            n.loads = n.loads.assign(carrier='load')
        d = -(pd.concat([(n.pnl(c).p.T.mul(n.df(c).sign, axis=0)
                .assign(carrier=n.df(c).carrier, bus=n.df(c).bus)
                .groupby(['bus', 'carrier']).sum().T
                .where(lambda x: x < 0)) for c in components], axis=1)
                .rename_axis(['bus', 'carrier'], axis=1))
        n.loads = n.loads.drop(columns='carrier')
        n.buses_t.p_minus_per_carrier = DataArray(d, dims=['snapshot','d'])\
                                                  .unstack('d')
    return n.buses_t.p_minus_per_carrier.reindex({'snapshot': snapshots})


def self_consumption(n, snapshots=None, override=False):
    """
    Inspection for self consumed power, i.e. power that is not injected in the
    network and consumed by the bus itself
    """
    if snapshots is None:
        snapshots = n.snapshots.rename('snapshot')
    if 'p_self' not in n.buses_t or override:
        n.buses_t.p_self = (xr.concat([power_production(n, n.snapshots),
                                       power_demand(n, n.snapshots)], 'bus')
                            .groupby('bus').min())
    return n.buses_t.p_self.reindex({'snapshot': snapshots})


