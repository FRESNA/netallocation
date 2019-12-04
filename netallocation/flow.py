#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:14:49 2018

@author: fabian
"""

# This side-package is created for use as flow and cost allocation.

from pypsa.descriptors import Dict
import pandas as pd
import xarray as xr
import numpy as np
from collections import Iterable
import os
from numpy import sign, conj, real
import logging
logger = logging.getLogger(__name__)

from .grid import (self_consumption, power_demand, power_production,
                        network_injection, network_flow, Incidence,
                        branch_inflow, branch_outflow,
                        PTDF, CISF, admittance, voltage, Ybus)
from .linalg import diag, inv, pinv
from .utils import parmap, upper, lower, last_to_first_level

def average_participation(n, snapshot, per_bus=False, normalized=False,
                          downstream=True, branch_components=['Line', 'Link'],
                          aggregated=True):
    """
    Allocate the network flow in according to the method 'Average
    participation' or 'Flow tracing' firstly presented in [1,2].
    The algorithm itself is derived from [3]. The general idea is to
    follow active power flow from source to sink (or sink to source)
    using the principle of proportional sharing and calculate the
    partial flows on each line, or to each bus where the power goes
    to (or comes from).

    This method provdes two general options:
        Downstream:
            The flow of each nodal power injection is traced through
            the network and decomposed the to set of lines/buses
            on which is flows on/to.
        Upstream:
            The flow of each nodal power demand is traced
            (in reverse direction) through the network and decomposed
            to the set of lines/buses where it comes from.

    [1] J. Bialek, “Tracing the flow of electricity,”
        IEE Proceedings - Generation, Transmission and Distribution,
        vol. 143, no. 4, p. 313, 1996.
    [2] D. Kirschen, R. Allan, G. Strbac, Contributions of individual
        generators to loads and flows, Power Systems, IEEE
        Transactions on 12 (1) (1997) 52–60. doi:10.1109/59.574923.
    [3] J. Hörsch, M. Schäfer, S. Becker, S. Schramm, and M. Greiner,
        “Flow tracing as a tool set for the analysis of networked
        large-scale renewable electricity systems,” International
        Journal of Electrical Power & Energy Systems,
        vol. 96, pp. 390–397, Mar. 2018.



    Parameters
    ----------
    network : pypsa.Network() object with calculated flow data

    snapshot : str
        Specify snapshot which should be investigated. Must be
        in network.snapshots.
    per_bus : Boolean, default True
        Whether to return allocation on buses. Allocate to lines
        if False.
    normalized : Boolean, default False
        Return the share of the source (sink) flow
    downstream : Boolean, default True
        Whether to use downstream or upstream method.

    """
    lower = lambda df: df.clip(upper=0)
    upper = lambda df: df.clip(lower=0)


    f0 = network_flow(n, snapshot, branch_components)
    f1 = network_flow(n, snapshot, branch_components, ingoing=False)
    f_in = f0.where(f0 > 0, - f1)
    f_out = f0.where(f0 < 0,  - f1)

    p = network_injection(n, snapshot, branch_components).T
    if aggregated:
        p_in = p.clip(lower=0)  # nodal inflow
        p_out = - p.clip(upper=0)  # nodal outflow
    else:
        p_in = power_production(n, [snapshot]).loc[snapshot]
        p_out = power_demand(n, [snapshot]).loc[snapshot]

    K = Incidence(n, branch_components)

    K_dir = K @ diag(sign(f_in))

#    Tau = lower(K_loss_dir) * f @ K.T + diag(p_in)

    Q = inv(lower(K_dir) @ diag(f_out) @ K.T + diag(p_in), pre_clean=True) \
            @ diag(p_in)
    R = inv(upper(K_dir) @ diag(f_in) @ K.T + diag(p_out), pre_clean=True) \
            @ diag(p_out)


    if not normalized and per_bus:
        Q = diag(p_out) @ Q
        R = diag(p_in) @ R
        if aggregated:
            # add self-consumption
            Q += diag(self_consumption(n, snapshot))
            R += diag(self_consumption(n, snapshot))

    q = (Q.rename_axis('in').rename_axis('source', axis=1)
         .replace(0, np.nan)
         .stack().swaplevel(0)
         .rename('upstream'))

    r = (R.rename_axis('out').rename_axis('sink', axis=1)
         .replace(0, np.nan)
         .stack()
         .rename('downstream'))

    T = (pd.concat([q,r], axis=0, keys=['upstream', 'downstream'],
                   names=['method', 'source', 'sink']).rename('allocation'))
    T = pd.concat([T], keys=[snapshot], names=['snapshot'])

    if per_bus:
        if downstream is not None:
            T = T.loc[:, 'downstream'] if downstream else T.loc[:, 'upstream']
    else:
        f = f_in if downstream else f_out
        T = _ap_normalized_to_flow(T, n, branch_components, f=f,
                                  normalized=normalized)
    return T


def _ap_normalized_to_flow(allocation, n, branch_components, f=None,
                          downstream=True, normalized=False):
    """Helper function to extend normalized average participation bus-to-bus
    allocation to line flow allocation. This function was pulled out of the
    main function to enable it for multiple snapshots.
    """

    sns = allocation.index.unique('snapshot')
    q = allocation.loc[:, 'upstream']\
            .rename_axis(['snapshot', 'source', 'in']).rename('upstream')
    r = allocation.loc[:, 'downstream']\
            .rename_axis(['snapshot', 'out', 'sink']).rename('downstream')

    if f is None:
        if downstream:
            f = branch_inflow(n, sns, branch_components)
        else:
            f = branch_outflow(n, sns, branch_components)
    # add bus0 and  bus1 to index
    f = pd.concat([f, n.branches().loc[branch_components, ['bus0', 'bus1']]],
                   axis=1).rename_axis(index=['component', 'branch_i']) \
           .set_index(['bus0', 'bus1'], append=True)\
           .rename_axis(columns='snapshot').stack()\
           .pipe(last_to_first_level)

    # absolute flow with directions
    f_dir = pd.concat(
            [f[f > 0].rename_axis(index={'bus0':'in', 'bus1': 'out'}),
             f[f < 0].swaplevel()
                     .rename_axis(index={'bus0':'out', 'bus1': 'in'})])

    if normalized:
        f_dir = (f_dir.groupby(level=['snapshot', 'component', 'branch_i'])
                 .transform(lambda ds: ds/ds.abs().sum()))
    return ((q * f_dir).dropna() * r).dropna() \
            .droplevel(['in', 'out'])\
            .rename('allocation') \
            .reorder_levels(['snapshot', 'source',  'sink',
                             'component', 'branch_i'])


def _ap_normalized_to_dispatch(allocation, n, aggregated):
    """Helper function to scale normalized average participation bus-to-bus
    allocation to real dispatch values. This function was created
    to enable calculating multiple snapshots at once.
    """
    q = allocation.loc[:, 'downstream'].rename('upstream')
    sns = q.index.unique('snapshot')
    branch_components = ['Line', 'Link']
    if aggregated:
        p_in = network_injection(n, sns, branch_components=branch_components)\
                .clip(lower=0).T.rename_axis('source')\
                .unstack().rename('p_in')[lambda ds: ds!=0]
    else:
        p_in = power_production(n, sns).T.unstack().rename('p_in')

    T =  (q * p_in).dropna()

    if aggregated:
        # add self-consumption
        s = self_consumption(n, sns)
        s = s.set_axis(pd.MultiIndex.from_tuples(zip(s.columns, s.columns)),
                      axis=1, inplace=False) \
             .rename_axis(['source', 'sink'], axis=1)\
             .unstack().pipe(last_to_first_level) \
             .rename('selfconsumption')\
             [lambda ds: ds!=0]
        T = T.append(s).sort_index(level=0)
    return T.rename('allocation')



def marginal_participation(n, snapshot=None, q=0.5, normalized=False,
                           vip=False, branch_components=['Link', 'Line']):
    '''
    Allocate line flows according to linear sensitvities of nodal power
    injection given by the changes in the power transfer distribution
    factors (PTDF)[1-3]. As the method is based on the DC-approximation,
    it works on subnetworks only as link flows are not taken into account.
    Note that this method does not exclude counter flows. It return either a
    Virtual Injection Pattern

    [1] F. J. Rubio-Oderiz, I. J. Perez-Arriaga, Marginal pricing of
        transmission services: a comparative analysis of network cost
        allocation methods, IEEE Transactions on Power Systems 15 (1)
        (2000) 448–454. doi:10.1109/59.852158.
    [2] M. Schäfer, B. Tranberg, S. Hempel, S. Schramm, M. Greiner,
        Decompositions of injection patterns for nodal flow allocation
        in renewable electricity networks, The European Physical
        Journal B 90 (8) (2017) 144.
    [3] T. Brown, “Transmission network loading in Europe with high
        shares of renewables,” IET Renewable Power Generation,
        vol. 9, no. 1, pp. 57–65, Jan. 2015.


    Parameters
    ----------
    network : pypsa.Network() object with calculated flow data

    snapshot : str
        Specify snapshot which should be investigated. Must be
        in network.snapshots.
    q : float, default 0.5
        split between net producers and net consumers.
        If q is zero, only the impact of net load is taken into
        account. If q is one, only net generators are taken
        into account.
    vip : Boolean, default True
        Whether to return allocation on buses. Allocate to lines
        if False.
    normalized : Boolean, default False
        Return the share of te allocated flow per line.

    '''
    snapshot = n.snapshots[0] if snapshot is None else snapshot
    H = PTDF(n, branch_components=branch_components, snapshot=snapshot)
    K = Incidence(n, branch_components=branch_components)
    f = network_flow(n, snapshot, branch_components)
    p = K @ f
    p_plus = p.clip(lower=0)
    # unbalanced flow from positive injection:
    f_plus = H @ p_plus
    k_plus = (q * f - f_plus) / p_plus.sum()
    if normalized:
        F = H.add(k_plus, axis=0).mul(p, axis=1).div(f, axis=0).fillna(0)
    else:
        F = H.add(k_plus, axis=0).mul(p, axis=1).fillna(0)
    if vip:
        P = (K @ F).rename_axis(index='injection pattern', columns='bus')
        return P
    else:
        return F


def equivalent_bilateral_exchanges(n, snapshot=None, normalized=False,
                                   vip=False, q=0.5,
                                   branch_components=['Line', 'Link']):
    """
    Sequentially calculate the load flow induced by individual
    power sources in the network ignoring other sources and scaling
    down sinks. The sum of the resulting flow of those virtual
    injection patters is the total network flow. This method matches
    the 'Marginal participation' method for q = 1. Return either Virtual
    Injection Patterns if vip is set to True, or Virtual Flow Patterns.


    Parameters
    ----------
    network : pypsa.Network object with calculated flow data
    snapshot : str
        Specify snapshot which should be investigated. Must be
        in network.snapshots.
    vip : Boolean, default True
        Whether to return allocation on buses. Allocate to lines
        if False.
    normalized : Boolean, default False
        Return the share of te allocated flow per line, only effective when
        vip is False.

    """
    snapshot = n.snapshots[0] if snapshot is None else snapshot
    H = PTDF(n, branch_components=branch_components, snapshot=snapshot)
    K = Incidence(n, branch_components=branch_components)

    f = network_flow(n, snapshot, branch_components)
    p = K @ f
    p_plus = p.clip(lower=0).to_frame()
    p_minus = p.clip(upper=0).to_frame()
    gamma = p_plus.sum().sum()
    P = (q * (diag(p_plus) + p_minus @ p_plus.T / gamma)
         + (1 - q) * (diag(p_minus) - p_plus @ p_minus.T / gamma))\
        .rename_axis(index='injection pattern', columns='bus')
    if not vip:
        F = (H @ P).rename_axis(columns='bus')
        if normalized:
            F = F.div(f, axis=0)
        return F
    else:
        return P



def zbus_transmission(n, snapshot=None, linear=False, downstream=None,
                      branch_components=['Line', 'Transformer']):
    '''
    This allocation builds up on the method presented in [1]. However, we
    provide for non-linear power flow an additional DC-approximated
    modification, neglecting the series resistance r for lines.


    [1] A. J. Conejo, J. Contreras, D. A. Lima, and A. Padilha-Feltrin,
        “$Z_{\rm bus}$ Transmission Network Cost Allocation,” IEEE Transactions
        on Power Systems, vol. 22, no. 1, pp. 342–349, Feb. 2007.

    '''
    n.calculate_dependent_values()
    snapshot = n.snapshots[0] if snapshot is None else snapshot
    b_comps = branch_components


    K = Incidence(n, branch_components=b_comps)
    Y = Ybus(n, b_comps, linear=linear)  # Ybus matrix

    v = voltage(n, snapshot, linear=linear)

    H = PTDF(n, b_comps) if linear else CISF(n, b_comps)

    i = Y @ v
    f = network_flow(n, snapshot, b_comps)
    if downstream is None:
        v_ = K.abs().T @ v / 2
    elif downstream:
        v_ = upper(K @ diag(sign(f))).T @ v
    else:
        v_ = -lower(K @ diag(sign(f))).T @ v

    if linear:
#        i >> network_injection(n, snapshot, branch_components=b_comps)
        fun = lambda v_, i : H @ diag(i) # which is the same as mp with q=0.5
    else:
        (np.conj(i) * v).apply(np.real) >> n.buses_t.p.loc[snapshot].T
        fun = lambda v_, i : ( diag(v_) @ conj(H) @ diag(conj(i)) ).applymap(real)


    if isinstance(snapshot, pd.Timestamp):
        v_ = v_.to_frame(snapshot)
        i = i.to_frame(snapshot)
        snapshot = [snapshot]

    q = pd.concat((fun(v_[sn], i[sn]) for sn in snapshot), keys=snapshot,
                  names=['snapshot', 'component', 'branch_i'])\
            .rename_axis(columns='bus')

    return q.stack()\
            .reorder_levels(['snapshot', 'bus', 'component', 'branch_i'])



def with_and_without_transit(n, snapshots=None,
                             branch_components=['Line', 'Link']):
    regions = n.buses.country.unique()

    if not n.links.empty:
        Y = pd.concat([admittance(n, branch_components, sn)
                       for sn in snapshots], axis=1,
                       keys=snapshots)
        def dynamic_subnetwork_PTDF(K, branches_i, snapshot):
            y = Y.loc[branches_i, snapshot].abs()
            return diag(y) @ K.T @ pinv(K @ diag(y) @ K.T)


    def regional_with_and_withtout_flow(region):
        in_region_buses = n.buses.query('country == @region').index
        vicinity_buses = pd.Index(
                            pd.concat(
                            [n.branches()[lambda df:
                                df.bus0.map(n.buses.country) == region].bus1,
                             n.branches()[lambda df:
                                 df.bus1.map(n.buses.country) == region].bus0]))\
                            .difference(in_region_buses)
        buses_i = in_region_buses.union(vicinity_buses).drop_duplicates()


        region_branches = n.branches()[lambda df:
                            (df.bus0.map(n.buses.country) == region) |
                            (df.bus1.map(n.buses.country) == region)] \
                            .rename_axis(['component', 'branch_i'])
        branches_i = region_branches.index

        K = Incidence(n, branch_components).loc[buses_i, branches_i]

        #create regional injection pattern with nodal injection at the border
        #accounting for the cross border flow
        f = pd.concat([n.pnl(c).p0.loc[snapshots].T for c in branch_components],
                      keys=branch_components, sort=True).reindex(branches_i)

        p = (K @ f)
        p.loc[in_region_buses] >> \
            network_injection(n, snapshots).loc[snapshots, in_region_buses].T

        #modified injection pattern without transition
        im = p.loc[vicinity_buses][lambda ds: ds > 0]
        ex = p.loc[vicinity_buses][lambda ds: ds < 0]

        largerImport_b = im.sum() > - ex.sum()
        scaleImport = (im.sum() + ex.sum()) / im.sum()
        scaleExport = (im.sum() + ex.sum()) / ex.sum()
        netImOrEx = (im * scaleImport).T\
                    .where(largerImport_b, (ex * scaleExport).T)
        p_wo = pd.concat([p.loc[in_region_buses], netImOrEx.T])\
                 .reindex(buses_i).fillna(0)

        if 'Link' not in f.index.unique('component'):
            y = admittance(n, ['Line'])[branches_i]
            H = diag(y) @ K.T @ pinv(K @ diag(y) @ K.T)
            f_wo = H @ p_wo
    #        f >> H @ p
        else:
            f_wo = pd.concat(
                    (dynamic_subnetwork_PTDF(K, branches_i, sn) @ p_wo[sn]
                        for sn in snapshots), axis=1, keys=snapshots)


        f, f_wo = f.T, f_wo.T
        return pd.concat([f, f_wo], axis=1, keys=['with', 'without'])
#        return {'flow': flow, 'loss': loss}
    flows = pd.concat((regional_with_and_withtout_flow(r) for r in regions),
                      axis=1, keys=regions,
                      names=['country', 'method', 'component', 'branch_i'])\
                .reorder_levels(['country', 'component', 'branch_i', 'method'],
                                axis=1).sort_index(axis=1)

    r_pu = n.branches().r_pu.fillna(0).rename_axis(['component', 'branch_i'])
    loss = (flows **2 * r_pu).sum(level=['country', 'method'], axis=1)
    return Dict({'flow': flows, 'loss': loss})


def marginal_welfare_contribution(n, snapshots=None, formulation='kirchhoff',
                                  return_networks=False):
    import pyomo.environ as pe
    from .opf import (extract_optimisation_results,
                      define_passive_branch_flows_with_kirchhoff)
    def fmap(f, iterable):
        # mapper for inplace functions
        for x in iterable:
            f(x)

    def profit_by_gen(n):
        price_by_generator = (n.buses_t.marginal_price
                              .reindex(columns=n.generators.bus)
                              .set_axis(n.generators.index, axis=1,
                                        inplace=False))
        revenue = price_by_generator * n.generators_t.p
        cost = n.generators_t.p.multiply(n.generators.marginal_cost, axis=1)
        return ((revenue - cost).rename_axis('profit')
                .rename_axis('generator', axis=1))

    if snapshots is None:
        snapshots = n.snapshots
    n.lopf(snapshots, solver_name='gurobi_persistent', formulation=formulation)
    m = n.model

    networks = {}
    networks['orig_model'] = n if return_networks else profit_by_gen(n)

    m.zero_flow_con = pe.ConstraintList()

    for line in n.lines.index:
#        m.solutions.load_from(n.results)
        n_temp = n.copy()
        n_temp.model = m
        n_temp.mremove('Line', [line])

        # set line flow to zero
        line_var = m.passive_branch_p['Line', line, :]
        fmap(lambda ln: m.zero_flow_con.add(ln == 0), line_var)

        fmap(n.opt.add_constraint, m.zero_flow_con.values())

        # remove cycle constraint from persistent solver
        fmap(n.opt.remove_constraint, m.cycle_constraints.values())

        # remove cycle constraint from model
        fmap(m.del_component, [c for c in dir(m) if 'cycle_constr' in c])
        # add new cycle constraint to model
        define_passive_branch_flows_with_kirchhoff(n_temp, snapshots, True)
        # add cycle constraint to persistent solver
        fmap(n.opt.add_constraint, m.cycle_constraints.values())

        # solve
        n_temp.results = n.opt.solve()
        m.solutions.load_from(n_temp.results)

        # extract results
        extract_optimisation_results(n_temp, snapshots,
                                     formulation='kirchhoff')

        if not return_networks:
            n_temp = profit_by_gen(n_temp)
        networks[line] = n_temp

        # reset model
        fmap(n.opt.remove_constraint, m.zero_flow_con.values())
        m.zero_flow_con.clear()

    return (pd.Series(networks)
            .rename_axis('removed line')
            .rename('Network'))



def flow_allocation(n, snapshots=None, method='Average participation',
                    parallelized=False, nprocs=None, to_hdf=False,
                    as_xarray=True, round_floats=8, **kwargs):
    """
    Function to allocate the total network flow to buses. Available
    methods are 'Average participation' ('ap'), 'Marginal
    participation' ('mp'), 'Virtual injection pattern' ('vip'),
    'Zbus transmission' ('zbus').



    Parameters
    ----------

    network : pypsa.Network object

    snapshots : string or pandas.DatetimeIndex
                (subset of) snapshots of the network

    per_bus : Boolean, default is False
              Whether to allocate the flow in an peer-to-peeer manner,

    method : string
        Type of the allocation method. Should be one of

            - 'Average participation'/'ap':
                Trace the active power flow from source to sink
                (or sink to source) using the principle of proportional
                sharing and calculate the partial flows on each line,
                or to each bus where the power goes to (or comes from).
            - 'Marginal participation'/'mp':
                Allocate line flows according to linear sensitvities
                of nodal power injection given by the changes in the
                power transfer distribution factors (PTDF)
            - 'Equivalent bilateral exchanges'/'ebe'
                Sequentially calculate the load flow induced by
                individual power sources in the network ignoring other
                sources and scaling down sinks.
            - 'Minimal flow shares'/'mfs'
            - 'Zbus transmission'/''zbus'


    Returns
    -------
    res : dict
        The returned dict consists of two values of which the first,
        'flow', represents the allocated flows within a mulitindexed
        pandas.Series with levels ['snapshot', 'bus', 'line']. The
        second object, 'cost', returns the corresponding cost derived
        from the flow allocation.
    """
    snapshots = n.snapshots if snapshots is None else snapshots
    snapshots = snapshots if isinstance(snapshots, Iterable) else [snapshots]
    if n.lines_t.p0.empty:
        raise ValueError('Flows are not given by the network, '
                         'please solve the network flows first')
    n.calculate_dependent_values()

    if method in ['Average participation', 'ap']:
        method_func = average_participation
    elif method in ['Marginal participation', 'mp']:
        method_func = marginal_participation
    elif method in ['Equivalent bilateral exchanges', 'ebe']:
        method_func = equivalent_bilateral_exchanges
    elif method in ['Zbus transmission', 'zbus']:
        method_func = zbus_transmission
    else:
        raise(ValueError('Method not implemented, please choose one out of'
                         "['Average participation',"
                           "'Marginal participation',"
                           "'Virtual injection pattern',"
                           "'Minimal flow shares',"
                           "'Zbus transmission']"))

    if snapshots is None:
        snapshots = n.snapshots
    if isinstance(snapshots, str):
        snapshots = [snapshots]

    if parallelized and not to_hdf:
        f = lambda sn: method_func(n, sn, **kwargs)
    else:
        def f(sn):
            if sn.is_month_start & (sn.hour == 0):
                logger.info('Allocating for %s %s'%(sn.month_name(), sn.year))
            return method_func(n, sn, **kwargs)


    if to_hdf:
        import random
        hash = random.getrandbits(12)
        store = '/tmp/temp{}.h5'.format(hash) if not isinstance(to_hdf, str) \
                else to_hdf
        periods = pd.period_range(snapshots[0], snapshots[-1], freq='m')
        p_str = lambda p: '_t_' + str(p).replace('-', '')
        for p in periods:
            p_slicer = snapshots.slice_indexer(p.start_time, p.end_time)
            gen = (f(sn) for sn in snapshots[p_slicer])
            pd.concat(gen).to_hdf(store, p_str(p))

        gen = (pd.read_hdf(store, p_str(p)) for p in periods)
        flow = pd.concat(gen)
        os.remove(store)

    elif parallelized:
        flow = pd.concat(parmap(f, snapshots, nprocs=nprocs))
    else:
        flow = pd.concat((f(sn) for sn in snapshots), keys=snapshots.rename('snapshot'))
    if round_floats is not None:
        flow = flow[flow.round(round_floats)!=0]
    if as_xarray:
        flow = xr.DataArray.from_series(flow.stack(), sparse=True).rename('allocation')
    return flow


