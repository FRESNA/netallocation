#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:18:23 2019

@author: fabian
"""

import pandas as pd
import numpy as np
from pypsa.pf import find_cycles as find_cycles
from .linalg import pinv, diag, null
import logging
logger = logging.getLogger(__name__)



def Incidence(n, branch_components=['Link', 'Line']):
    buses = n.buses.index
    return pd.concat([(n.df(c).assign(K=1).set_index('bus0', append=True)['K']
                     .unstack().reindex(columns=buses).fillna(0).T)
                     - (n.df(c).assign(K=1).set_index('bus1', append=True)['K']
                     .unstack().reindex(columns=buses).fillna(0).T)
                     for c in branch_components],
                     keys=branch_components, axis=1, sort=False)\
            .reindex(columns=n.branches().loc[branch_components].index)\
            .rename_axis(columns=['component', 'branch_i'])

def Cycles(n, dense=True, update=True):
    if (not 'C' in n.__dir__()) | update:
        find_cycles(n)
        if dense:
            branches_i = n.branches()["bus0"].index
            n.C = pd.DataFrame(n.C.todense(), index=branches_i)
        return n.C.T
    else:
        return n.C.T

def active_cycles(n, snapshot):
    # copy original links
    orig_links = n.links.copy()
    # modify current links
    n.links = n.links[n.links_t.p0.loc[snapshot].abs() >= 1e-8]
    C = Cycles(n, update=True)
    # reassign original links
    n.links = orig_links
    return C.reindex(columns=n.branches().index, fill_value=0)


def impedance(n, branch_components=['Line', 'Link'], snapshot=None,
              pu_system=True, linear=True):
    n.lines = n.lines.assign(carrier=n.lines.bus0.map(n.buses.carrier))

    #impedance
    x = 'x_pu' if pu_system else 'x'
    r = 'r_pu' if pu_system else 'r'

    if linear:
        _z = [n.lines[x].where(n.lines.carrier == 'AC', n.lines[r])]
        z = pd.concat(_z, keys=["Line"]).rename_axis(['component', 'branch_i'])
    else:
        z = n.branches().eval('r + 1.j * x')\
             .reindex(branch_components, level=0) \
             .rename_axis(['component', 'branch_i'])


    if ('Link' not in branch_components) | n.links.empty :
        return z
    if snapshot is None:
        logger.warn('Link in argument "branch_components", but no '
                        'snapshot given. Falling back to first snapshot')
        snapshot = n.snapshots[0]
    elif isinstance(snapshot, pd.DatetimeIndex):
        snapshot = snapshot[0]

    f = pd.concat([n.lines_t.p0.loc[snapshot], n.links_t.p0.loc[snapshot]],
                  keys=['Line', 'Link'])
    n.lines = n.lines.assign(carrier=n.lines.bus0.map(n.buses.carrier))

    C = active_cycles(n, snapshot)

    C_mix = C[((( C != 0) & (f != 0)).groupby(level=0, axis=1).any()).Link]

    if C_mix.empty:
        omega = f[['Link']]
    elif z.empty:
        omega = null(C_mix[['Link']] @ diag(f[['Link']]))[0]
    else:
        omega = - pinv(C_mix[['Link']] @ diag(f[['Link']])) \
                @ C_mix[['Line']] @ diag(z) @ f[['Line']]

    omega = omega.round(10) #numerical issues either
    omega[(omega == 0) & (f[['Link']] != 0)] = (1/f).fillna(0)
    return pd.concat([z, omega]).loc[branch_components]\
             .rename_axis(['component', 'branch_i'])


def admittance(n, branch_components=['Line', 'Link'], snapshot=None,
               pu_system=True, linear=True):
    return (1/impedance(n, branch_components, snapshot, pu_system=pu_system,
                        linear=linear))\
            .replace([np.inf, -np.inf], 0)


def PTDF(n, branch_components=['Line'], snapshot=None, pu_system=True,
         linear=True):
    n.calculate_dependent_values()
    K = Incidence(n, branch_components)
    y = admittance(n, branch_components, snapshot,
                   pu_system=pu_system, linear=linear)
    return diag(y) @ K.T @ pinv(K @ diag(y) @ K.T)


def Ybus(n, branch_components=['Line', 'Link'], snapshot=None,
         pu_system=True, linear=True):
    K = Incidence(n, branch_components)
    y = admittance(n, branch_components, snapshot,
                   pu_system=pu_system, linear=linear)
    if linear:
        return K @ diag(y) @ K.T

    def get_Ybus(sub):
        sub.calculate_PTDF()
        sub.calculate_Y()
        return pd.DataFrame(sub.Y.todense(), index=sub.buses_o,
                         columns=sub.buses_o)
    if not pu_system:
        raise NotImplementedError('Non per unit system '
                                  'for non-linear Ybus matrix not implemented')
    return pd.concat(n.sub_networks.obj.apply(get_Ybus).to_dict())



def Zbus(n, branch_components=['Line', 'Link'], snapshot=None,
         pu_system=True, linear=True):
    return pinv(Ybus(n, branch_components, snapshot,
                     pu_system=pu_system, linear=linear))


def network_flow(n, snapshots=None, branch_components=['Link', 'Line'],
                 ingoing=True):
    snapshots = n.snapshots if snapshots is None else snapshots
    p = 'p0' if ingoing else 'p1'
    return pd.concat([n.pnl(b)[p] for b in branch_components], axis=1,
                      keys=branch_components)\
             .rename_axis(['component', 'branch_i'], axis=1)\
             .loc[snapshots]


def network_injection(n, snapshots=None, branch_components=['Link', 'Line']):
    """
    Function to determine the total network injection including passive and
    active branches.
    """
    f = network_flow(n, snapshots, branch_components).T
    K = Incidence(n, branch_components)
    return (K @ f).T


def is_balanced(n, tol=1e-9):
    """
    Helper function to double check whether network flow is balanced
    """
    K = Incidence(n)
    F = pd.concat([n.lines_t.p0, n.links_t.p0], axis=1,
                  keys=['Line', 'Link']).T
    return (K.dot(F)).sum(0).max() < tol


def power_production(n, snapshots=None,
                     components=['Generator', 'StorageUnit'],
                     per_carrier=False, update=False):
    if snapshots is None:
        snapshots = n.snapshots.rename('snapshot')
    if 'p_plus' not in n.buses_t or update:
        n.buses_t.p_plus = (sum(n.pnl(c).p
                            .mul(n.df(c).sign).T
                            .clip(lower=0)
                            .assign(bus=n.df(c).bus)
                            .groupby('bus').sum()
                            .reindex(index=n.buses.index, fill_value=0).T
                            for c in components)
                            .rename_axis('source', axis=1))
    if 'p_plus_per_carrier' not in n.buses_t or update:
        n.buses_t.p_plus_per_carrier = (
                pd.concat([(n.pnl(c).p.T
                            .assign(carrier=n.df(c).carrier, bus=n.df(c).bus)
                            .groupby(['bus', 'carrier']).sum().T
                            .where(lambda x: x > 0))
                          for c in components], axis=1)
                .rename_axis(['source', 'sourcetype'], axis=1))

    if per_carrier:
        return n.buses_t.p_plus_per_carrier.reindex(snapshots)
    return n.buses_t.p_plus.reindex(snapshots)


def power_demand(n, snapshots=None,
                 components=['Load', 'StorageUnit'],
                 per_carrier=False, update=False):
    if snapshots is None:
        snapshots = n.snapshots.rename('snapshot')
    if 'p_minus' not in n.buses_t or update:
        n.buses_t.p_minus = (sum(n.pnl(c).p.T
                             .mul(n.df(c).sign, axis=0)
                             .clip(upper=0)
                             .assign(bus=n.df(c).bus)
                             .groupby('bus').sum()
                             .reindex(index=n.buses.index, fill_value=0).T
                             for c in components).abs()
                             .rename_axis('sink', axis=1))

    if 'p_minus_per_carrier' not in n.buses_t or update:
        if components == ['Generator', 'StorageUnit']:
            intersc = (pd.Index(n.storage_units.carrier.unique())
                       .intersection(pd.Index(n.generators.carrier.unique())))
            assert (intersc.empty), (
                    'Carrier names {} of compoents are not unique'
                    .format(intersc))
        n.loads = n.loads.assign(carrier='load')
        n.buses_t.p_minus_per_carrier = -(
                pd.concat([(n.pnl(c).p.T
                .mul(n.df(c).sign, axis=0)
                .assign(carrier=n.df(c).carrier, bus=n.df(c).bus)
                .groupby(['bus', 'carrier']).sum().T
                .where(lambda x: x < 0)) for c in components], axis=1)
                .rename_axis(['sink', 'sinktype'], axis=1))
        n.loads = n.loads.drop(columns='carrier')

    if per_carrier:
        return n.buses_t.p_minus_per_carrier.reindex(snapshots)
    return n.buses_t.p_minus.reindex(snapshots)


def self_consumption(n, snapshots=None, override=False):
    """
    Inspection for self consumed power, i.e. power that is not injected in the
    network and consumed by the bus itself
    """
    if snapshots is None:
        snapshots = n.snapshots.rename('snapshot')
    if 'p_self' not in n.buses_t or override:
        n.buses_t.p_self = (pd.concat([power_production(n, n.snapshots),
                                       power_demand(n, n.snapshots)], axis=1)
                            .groupby(level=0, axis=1).min())
    return n.buses_t.p_self.loc[snapshots]


