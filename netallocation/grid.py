#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module comprises all electricity grid relevant functions.
"""

import pandas as pd
import xarray as xr
from xarray import DataArray
import numpy as np
from .linalg import pinv, diag, null, dot
from .utils import (get_branches_i, reindex_by_bus_carrier, check_branch_comps,
                    check_snapshots, check_passive_branch_comps,
                    snapshot_weightings)
from sparse import as_coo
import logging
import networkx as nx
import scipy
logger = logging.getLogger(__name__)


def Incidence(n, branch_components=None, sparse=False):
    """
    Calculate the Incidence matrix for a given networ with given branch components.

    Parameters
    ----------
    n : pypsa.Netowrk
    branch_components : list, optional
        List of branch components to be included in the Incidence matris.
        The default is None results in n.branch_components.
    sparse : bool, optional
        Whether the resulting data should be sparse or not. The default is False.

    Returns
    -------
    K : xr.DataArray
        Incidence matrix with dimensions N (#buses) x L (#branches).

    """
    branch_components = check_branch_comps(branch_components, n)
    if sparse:
        K = as_coo(n.incidence_matrix(branch_components))
    else:
        K = n.incidence_matrix(branch_components).todense()
    branches_i = get_branches_i(n, branch_components)
    return DataArray(K, coords=(n.buses.index, branches_i), dims=['bus', 'branch'])


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
        branches = branches.reindex(branches_i)
    branches = branches.assign(index = branches_i)
    branches_bus0 = branches['source']
    mgraph = nx.from_pandas_edgelist(branches, edge_attr=True,
                                     create_using=nx.MultiGraph)
    graph = nx.OrderedGraph(mgraph)
    cycles = nx.cycle_basis(graph)
    #number of 2-edge cycles
    num_multi = len(mgraph.edges()) - len(graph.edges())
    C = scipy.sparse.dok_matrix((len(branches_bus0), len(cycles) + num_multi))
    for j,cycle in enumerate(cycles):
        for i, start in enumerate(cycle):
            end = cycle[(i+1)%len(cycle)]
            branch = branches_i.get_loc(graph[start][end]['index'])
            sign = +1 if branches_bus0.iat[branch] == cycle[i] else -1
            C[branch, j] += sign
    #counter for multis
    c = len(cycles)
    #add multi-graph 2-edge cycles for multiple branches between same pairs of buses
    for u,v in graph.edges():
        bs = list(mgraph[u][v].values())
        if len(bs) > 1:
            first = branches_i.get_loc(bs[0]['index'])
            for b in bs[1:]:
                other = branches_i.get_loc(b['index'])
                sign = -1 if branches_bus0.iat[other] == branches_bus0.iat[first] else +1
                C[first, c] = 1
                C[other, c] = sign
                c+=1
    return DataArray(C.todense(),  {'branch': branches_i, 'cycle': range(C.shape[1])},
                    ('branch', 'cycle'))

def impedance(n, branch_components=None, snapshot=None,
              pu_system=True, linear=True, skip_pre=False):
    """
    Calculate the impedance of the network branches.

    Naturally the impdance of controllable branches is not existent. However,
    in https://www.preprints.org/manuscript/202001.0352/v1 a method was
    presented how to calculate the impendance of controllable branches if they
    were passive AC lines. If 'Link' is included in branch_components,
    the flow-dependent pseudo-impedance is calculated based on the formulation
    presented in the paper. Note that in this case the flow must be given
    for all branches.

    Parameters
    ----------
    n : pypsa.Network
    branch_components : list, optional
        List of branch components. The default None results in
        n.passive_branch_components.
    snapshot : str/pd.Timestamp, optional
        Only relevant if 'Link' in branch_components. The default None results
        in the first snapshot of n.
    pu_system : bool, optional
        Whether the use the per uni system for the impendance.
        The default is True.
    linear : bool, optional
        Whether to use the linear approximation. The default is True.
    skip_pre : bool, optional
        Whether to calcuate dependent quantities beforehand. The default is False.

    Returns
    -------
    z : xr.DataArray
        Impedance for each branch in branch_components.

    """
    #standard impedance, note z must not be inf or nan
    branch_components = check_passive_branch_comps(branch_components, n)
    x = 'x_pu' if pu_system else 'x'
    r = 'r_pu' if pu_system else 'r'

    if not skip_pre:
        if pu_system and (n.lines[x] == 0).all():
            n.calculate_dependent_values()

    comps = sorted(set(branch_components) & n.passive_branch_components)
    if linear:
        z = pd.concat({c: n.df(c)[x].where(n.df(c).bus0.map(n.buses.carrier) == 'AC',
                    n.df(c)[r]) for c in comps})
    else:
        z = pd.concat({c: n.df(c).eval(f'{r} + 1.j * {x}') for c in comps})
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
    branches_i = f.get_index('branch')
    C = Cycles(n, branches_i[abs(f).values > 1e-8])\
            .reindex(branch=branches_i, fill_value=0)
    # C_mix is all the active cycles where at least one link is included
    C_mix = C[:, ((C != 0) & (f != 0)).groupby('component').any().loc['Link'].values]

    if not C_mix.size:
        sub = f.loc['Link'][abs(f.loc['Link']).values > 1e-8]
        omega = DataArray(1, sub.coords)
    elif not z.size:
        omega = null(C_mix.loc['Link'] * f.loc['Link'])[0]
    else:
        d = {'branch': 'Link'}
        omega = - dot(pinv(dot(C_mix.loc['Link'].T, diag(f.loc['Link']))),
           dot(C_mix.drop_sel(d).T, diag(z), f.drop_sel(d)))

    omega = omega.round(10).assign_coords({'component':'Link'})
    omega[(omega == 0) & (f.loc['Link'] != 0)] = 1
    Z = z.reindex_like(f).copy()
    Z.loc['Link'] = omega.reindex_like(f.loc['Link'], fill_value=0)
    return Z.assign_coords(snapshot=snapshot)

def admittance(n, branch_components=None, snapshot=None,
               pu_system=True, linear=True):
    """
    Calculate the series admittance. This is the inverse of the impedance,
    see :func:`impedance` for further information.
    """
    y = 1/impedance(n, branch_components, snapshot, pu_system, linear)
    return y.where(~np.isinf(y), 0)


def series_shunt_admittance(n, pu_system=True, branch_components=None):
    "Get the series shunt admittance."
    branch_components = check_passive_branch_comps(branch_components, n)
    g, b = ('g_pu', 'b_pu') if pu_system else ('g', 'b')
    shunt = pd.concat({c.name: c.df[g].fillna(0) + 1.j * c.df[b].fillna(0)
                       for c in n.iterate_components(branch_components)})
    return DataArray(shunt.rename_axis(['component', 'branch_i']), dims='branch')


def shunt_admittance(n, pu_system=True, branch_components=None):
    """
    Get the total nodal shunt admittance. This comprises all series shunt
    admittance of adjacent branches.
    """
    branch_components = check_passive_branch_comps(branch_components, n)
    g, b = ('g_pu', 'b_pu') if pu_system else ('g', 'b')
    K = Incidence(n, branch_components=branch_components)
    series_shunt = series_shunt_admittance(n, pu_system)
    nodal_shunt = n.shunt_impedances.fillna(0).groupby('bus').sum()\
                    .eval(f'{g} + 1.j * {b}')\
                    .reindex(n.buses.index, fill_value=0)
    nodal_shunt = DataArray(nodal_shunt, dims='bus')
    return 0.5 * abs(K) @ series_shunt + nodal_shunt

def PTDF(n, branch_components=None, snapshot=None, pu_system=True, update=True):
    """
    Calculate the Power Tranfer Distribution Factors (PTDF)

    If branch_component and snapshots is None (default), they are set to
    n.branch_components and n.snapshots respectively.

    If branch_components includes 'Link' the time-dependent PTDF matrix is
    calculated on the basis of the flow-dependent pseudo-impedance (only works
    for solved networks, see :func:`impedance`).
    """
    branch_components = check_branch_comps(branch_components, n)
    if 'Link' in branch_components or update or '_ptdf' not in n.__dir__():
        n.calculate_dependent_values()
        K = Incidence(n, branch_components)
        y = admittance(n, branch_components, snapshot, pu_system=pu_system)
        n._ptdf = dot(y * K.T, pinv(dot(K * y, K.T)))
        if snapshot is not None:
            n._ptdf = n._ptdf.assign_coords(snapshot=snapshot)
    return n._ptdf


def CISF(n, branch_components=None, pu_system=True):
    branch_components = check_passive_branch_comps(branch_components, n)
    n.calculate_dependent_values(), n.determine_network_topology()
    K = Incidence(n, branch_components)
    y = admittance(n, branch_components, pu_system=pu_system, linear=False)
    Z = Zbus(n, branch_components, pu_system=pu_system, linear=False)
    return dot(diag(y), K.T, Z)

def Ybus(n, branch_components=None, snapshot=None, pu_system=True, linear=True):
    """
    Calculate the Ybub matrix (or weighited Laplacian) for a given network for
    given branch_components.

    If branch_component and snapshots is None (default), they are set to
    n.branch_components and n.snapshots respectively.

    If 'Link' is included in branch_components, then their weightings are
    derived from their current pseudo-impedance dependent on the
    current flow (see :func:`impedance`).
    """
    branch_components = check_passive_branch_comps(branch_components, n)
    K = Incidence(n, branch_components)
    y = admittance(n, branch_components, snapshot, pu_system=pu_system,
                   linear=linear)
    Y = dot(K, diag(y), K.T)
    if linear:
        return Y
    else:
        if not pu_system:
            raise NotImplementedError('Non per unit system '
                                      'for non-linear Ybus matrix not '
                                      'implemented')
        return Y + diag(shunt_admittance(n, pu_system).sel(bus=Y.bus)).values

        def get_Ybus(sub):
            sub.calculate_PTDF()
            sub.calculate_Y()
            return pd.DataFrame(sub.Y.todense(), index=sub.buses_o,
                             columns=sub.buses_o)
        return n.sub_networks.obj.apply(get_Ybus)


def Zbus(n, branch_components=None, snapshot=None,
         pu_system=True, linear=True):
    """
    Calculate the Zbus matrix for given branch_components.

    If branch_component and snapshots is None (default), they are set to
    n.branch_components and n.snapshots respectively.

    If branch_components includes 'Link' the time-dependent Zbus matrix is
    calculated on the basis of the flow-dependent pseudo-impedance (only works
    for solved networks, see :func:`impedance`).
    """
    return pinv(Ybus(n, branch_components=branch_components,
                     snapshot=snapshot,
                     pu_system=pu_system, linear=linear))

def voltage(n, snapshots=None, linear=True, pu_system=True):
    """
    Get the voltage at each bus of a solved network for given snapshots.

    If snapshots is None (default), all n.snapshots are taken.
    """
    snapshots = check_snapshots(snapshots, n)
    if linear:
        v = n.buses_t.v_ang.loc[snapshots] + 1
#        v = np.exp(- 1.j * n.buses_t.v_ang).T[snapshots]
    else:
        v = (n.buses_t.v_mag_pu * np.exp(1.j * n.buses_t.v_ang)).loc[snapshots]

    if not pu_system:
        v *= n.buses.v_nom

    if isinstance(snapshots, (list, pd.Index)):
        return DataArray(v.T.reindex(n.buses.index), dims=['bus', 'snapshot'])
    else:
        return DataArray(v.reindex(n.buses.index), dims='bus')\
                .assign_coords(snapshot=snapshots)


def network_flow(n, snapshots=None, branch_components=None, ingoing=True,
                 linear=True):
    """
    Get the flow of the network on given branch_components for given snapshots.

    If branch_component and snapshots is None (default), they are set to
    n.branch_components and n.snapshots respectively.
    """
    snapshots = check_snapshots(snapshots, n)
    comps = check_branch_comps(branch_components, n)
    p = 'p0' if ingoing else 'p1'
    axis = int(isinstance(snapshots, (list, pd.Index)))
    f = pd.concat({b: n.pnl(b)[p].loc[snapshots, n.df(b).index] for b in comps},
                  axis=axis)
    if not linear:
        q = 'q0' if ingoing else 'q1'
        pcomps = sorted(set(branch_components) & n.passive_branch_components)
        fq = pd.concat({b: n.pnl(b)[q].loc[snapshots, n.df(b).index] for b in pcomps},
                         axis=axis)
        f = f.add(1.j * fq, fill_value=0)
    f = f.rename_axis(['component', 'branch_i'], axis=axis)
    if axis:
        return DataArray(f.T, dims=['branch', 'snapshot'])
    else:
        return DataArray(f, dims='branch').assign_coords(snapshot=snapshots)


def branch_inflow(n, snapshots=None, branch_components=None, linear=True):
    """
    Calculate the flow that goes into a branch.

    If branch_component and snapshots is None (default), they are set to
    n.branch_components and n.snapshots respectively.
    """
    f0 = network_flow(n, snapshots, branch_components, linear=linear).T
    f1 = network_flow(n, snapshots, branch_components, False, linear).T
    return f0.where(f0 > 0, - f1)


def branch_outflow(n, snapshots=None, branch_components=None, linear=True):
    """
    Determine the flow that comes out of a branch.

    If branch_component and snapshots is None (default), they are set to
    n.branch_components and n.snapshots respectively.
    """
    f0 = network_flow(n, snapshots, branch_components, linear=linear).T
    f1 = network_flow(n, snapshots, branch_components, False, linear).T
    return f0.where(f0 < 0,  - f1)


def network_injection(n, snapshots=None, branch_components=None, linear=True):
    """
    Determine the total network injection including passive and active branches.

    If branch_component and snapshots is None (default), they are set to
    n.branch_components and n.snapshots respectively.
    """
    f0 = network_flow(n, snapshots, branch_components, linear=linear).T
    f1 = network_flow(n, snapshots, branch_components, False, linear).T
    K = Incidence(n, branch_components)
    return (K.clip(min=0) @ f0 - K.clip(max=0) @ f1).T


def _one_port_attr(n, snapshots=None, attr='p', comps=None):
    """
    Retrieve a time-dependent attribute for given component, indexed by bus
    and carrier.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : (subset of) n.snashots
    attr : str
        Attribute which should be grouped per bus and carriert.
        The default is 'p'.

    Returns
    -------
    xr.DataArray

    """
    snapshots = check_snapshots(snapshots, n)
    if comps is None:
        comps = [c for c in sorted(n.one_port_components) if not n.df(c).empty]
    gen = (reindex_by_bus_carrier(n.pnl(c)[attr].loc[snapshots] * n.df(c).sign,
                                  c, n) for c in comps)
    return xr.concat(gen, dim='carrier', fill_value=0)

def power_production(n, snapshots=None, per_carrier=False, update=False):
    """
    Calculate the gross power production per bus and optionally carrier.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots, default None
        If None, all snapshots are taken.
    per_carrier : bool, optional
        Whether to calculate the power production per bus and carrier.
        The default is False.
    update : bool, optional
        Whether to recalculate cashed data. The default is False.

    Returns
    -------
    prod : xr.DataArray
        Power production data with dimensions snapshot, bus, carrier (optionally).

    """
    snapshots = check_snapshots(snapshots, n)
    if 'p_plus' not in n.buses_t or update:
        prod = _one_port_attr(n, n.snapshots, attr='p')
        n.buses_t['p_plus'] = prod.sel(carrier=(prod>=1e-8).any(['snapshot', 'bus']))\
                                  .clip(min=0)
    prod = n.buses_t.p_plus.sel(snapshot=snapshots)
    if not per_carrier:
        prod = prod.sum('carrier')
    return prod.reindex(bus=n.buses.index, fill_value=0)

def energy_production(n, snapshots=None, per_carrier=False, update=False):
    """
    Calculate the gross energy production per bus (and carrier).


    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots, default None
        If None, all snapshots are taken.
    per_carrier : bool, optional
        Whether to calculate the power production per bus and carrier.
        The default is False.
    update : bool, optional
        Whether to recalculate cashed data. The default is False.

    Returns
    -------
    Produced energy data with dimensions snapshot, bus, carrier (optionally).

    """
    snapshots = check_snapshots(snapshots, n)
    sn_weightings = DataArray(n.snapshot_weightings.loc[snapshots], dims='snapshot')
    return power_production(n, snapshots, per_carrier, update) * sn_weightings

def power_demand(n, snapshots=None, per_carrier=False, update=False):
    """
    Calculate the gross power consumption per bus and optionally carrier.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots, default None
        If None, all snapshots are taken.
    per_carrier : bool, optional
        Whether to calculate the power demand per bus and carrier.
        The default is False.
    update : bool, optional
        Whether to recalculate cashed data. The default is False.

    Returns
    -------
    prod : xr.DataArray
        Power demand data with dimensions snapshot, bus, carrier (optionally).

    """
    snapshots = check_snapshots(snapshots, n)
    if 'p_minus' not in n.buses_t or update:
        demand = _one_port_attr(n, n.snapshots)
        n.buses_t['p_minus'] = (- demand).sel(carrier=(demand<=-1e-8)
                                          .any(['snapshot', 'bus'])).clip(min=0)
    demand = n.buses_t.p_minus.sel(snapshot=snapshots)
    if not per_carrier:
        demand = demand.sum('carrier')
    return demand.reindex(bus=n.buses.index, fill_value=0)


def energy_demand(n, snapshots=None, per_carrier=False, update=False):
    """
    Calculate the gross energy consumption per bus and optionally carrier.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots, default None
        If None, all snapshots are taken.
    per_carrier : bool, optional
        Whether to calculate the power demand per bus and carrier.
        The default is False.
    update : bool, optional
        Whether to recalculate cashed data. The default is False.

    Returns
    -------
    Energy demand data with dimensions snapshot, bus, carrier (optionally).

    """
    snapshots = check_snapshots(snapshots, n)
    sn_weightings = snapshot_weightings(n, snapshots)
    return power_demand(n, snapshots, per_carrier, update) * sn_weightings


def self_consumption(n, snapshots=None, update=False):
    """
    Calculate the self consumed power, i.e. power that is not injected in the
    network and consumed by the bus itself
    """
    snapshots = check_snapshots(snapshots, n)
    if 'p_self' not in n.buses_t or update:
        n.buses_t.p_self = (xr.concat(
            [power_production(n, n.snapshots), power_demand(n, n.snapshots)], 'bus')
            .groupby('bus').min().reindex(bus=n.buses.index, fill_value=0))
    return n.buses_t.p_self.loc[snapshots]

