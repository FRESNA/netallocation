#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:14:49 2018

@author: fabian
"""

# This side-package is created for use as flow and cost allocation.

from .linalg import dot
from pypsa.descriptors import Dict
import pandas as pd
import xarray as xr
from xarray import DataArray, Dataset
import numpy as np
from numpy import sign, conj, real
import logging
logger = logging.getLogger(__name__)

from .grid import (self_consumption, power_demand, power_production,
                        network_injection, network_flow, Incidence,
                        branch_inflow, branch_outflow,
                        PTDF, CISF, admittance, voltage, Ybus)
from .linalg import diag, inv, pinv, dedup_axis
from .utils import parmap, upper, lower


def average_participation(n, snapshot, dims=['source', 'sink'],
                    branch_components=None, aggregated=True, downstream=True,
                    include_self_consumption=True):
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
    dims : list or string
        list of dimensions to be included, if set to "all", the full dimensions
        are calculated ['source', 'branch', 'sink']
    downstream : Boolean, default True
        Whether to use downstream or upstream method for performing the
        flow-tracing.
    aggregated: boolean, defaut True
        Within the aggregated coupling scheme (obtained if set to True),
        power production and demand are 'aggregated' within the corresponding
        bus. Therefore only the net surplus or net deficit of a bus is
        allocated to other buses.
        Within the direct coupling scheme (if set to False), production and
        demand are considered independent of the bus, therefore the power
        production and demand are allocated to all buses at the same time.
        Even if a bus has net deficit, its power production can be
        allocated to other buses.
    include_self_consumption: boolean, default True
        Whether to include self consumption of each buses within the aggregated
        coupling scheme.
    """
    dims = ['source', 'branch', 'sink'] if dims == 'all' else dims

    f0 = network_flow(n, snapshot, branch_components)
    f1 = network_flow(n, snapshot, branch_components, ingoing=False)
    f_in = f0.where(f0 > 0, - f1)
    f_out = f0.where(f0 < 0,  - f1)
    p = network_injection(n, snapshot, branch_components)

    filter_null = lambda da, dim: da.where(da != 0).dropna(dim, how='all')

    if aggregated:
        # nodal inflow and nodal outflow
        p_in = upper(p).rename(bus='source')#.pipe(filter_null, 'source')
        p_out = - lower(p).rename(bus='sink')#.pipe(filter_null, 'sink')
    else:
        p_in = power_production(n, [snapshot]).loc[snapshot].rename(bus='source')
        p_out = power_demand(n, [snapshot]).loc[snapshot].rename(bus='sink')

    K = Incidence(n, branch_components)
    K_dir = K * sign(f_in)

    J = inv(dot(lower(K_dir), diag(f_out), K.T) + np.diag(p_in), True)
    Q = J.pipe(dedup_axis, ('sink', 'source')) * p_in
    J = inv(dot(upper(K_dir), diag(f_in), K.T) + np.diag(p_out), True)
    R = J.pipe(dedup_axis, ('source', 'sink')) * p_out

    Q = filter_null(Q, 'source')
    R = filter_null(R, 'sink')

    if downstream:
        A, kind = Q * p_out, 'downstream'
    else:
        A, kind = R * p_in, 'upstream'

    if aggregated and include_self_consumption:
        selfcon = self_consumption(n, snapshot)
        A += diag(selfcon, ('source', 'sink')).reindex_like(A)

    res = A.to_dataset(name='peer_to_peer').assign_attrs(method='Average Participation')

    if 'branch' in dims:
        f = f_in if downstream else f_out
        T = dot(diag(f), upper(K_dir.T), Q.fillna(0)) * \
            dot(lower(K_dir.T), -R.fillna(0))
        T = T.assign_attrs(kind=kind)
        res = res.assign({'peer_on_branch_to_peer': T})
    return res



def marginal_participation(n, snapshot=None, q=0.5, branch_components=None):
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

    '''
    snapshot = n.snapshots[0] if snapshot is None else snapshot
    H = PTDF(n, branch_components=branch_components, snapshot=snapshot)
    K = Incidence(n, branch_components=branch_components)
    f = network_flow(n, snapshot, branch_components)
    p = K @ f
    p_plus = upper(p)
    # unbalanced flow from positive injection:
    f_plus = H @ p_plus
    k_plus = (q * f - f_plus) / p_plus.sum()
    F = (H + k_plus) * p
#    pattr = {'dimension 0': 'bus', 'dimension 1': 'injection pattern'}
    P = dot(K, F).pipe(dedup_axis, ('bus', 'injection_pattern'))
    res = Dataset({'virtual_injection_pattern': P, 'virtual_flow_pattern': F},
                  attrs={'method': 'Marginal Participation'})
    return res


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
    f = network_flow(n, [snapshot], branch_components)
    p = K @ f
    p_plus = upper(p)
    p_minus = lower(p)
    p_pl = p_plus.loc[:, snapshot] # same as one-dimensional
    p_min = p_minus.loc[:, snapshot]
    A = dot(p_minus, p_plus.T) / float(p_pl.sum())
    B = dot(p_plus, p_minus.T) / float(p_pl.sum())
    new_dims = ('bus', 'injection_pattern')
    P = (q * (dedup_axis(A, new_dims) + diag(p_pl, new_dims))
         + (q - 1) * (dedup_axis(B, new_dims) - diag(p_min, new_dims)) )
    F = (H @ P).rename(injection_pattern='bus')
    res = Dataset({'virtual_injection_pattern': P, 'virtual_flow_pattern': F},
                  attrs={'method': 'Eqivalent Bilateral Exchanges'})
    return res



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

func_dict = {'Average participation': average_participation,
             'ap': average_participation,
             'Marginal participation': marginal_participation,
             'mp': marginal_participation,
             'Equivalent bilateral exchanges': equivalent_bilateral_exchanges,
             'ebe': equivalent_bilateral_exchanges,
             'Zbus transmission': zbus_transmission,
             'zbus': zbus_transmission}

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
    if all(c.pnl.p0.empty for c in n.iterate_components(n.branch_components)):
        raise ValueError('Flows are not given by the network, please solve the '
                         'network flows first')
    n.calculate_dependent_values()

    if method not in func_dict.keys():
        raise(ValueError('Method not implemented, please choose one out of'
                         f'{list(func_dict.keys())}'))

    if isinstance(snapshots, (str, pd.Timestamp)):
        return func_dict[method](n, snapshots, **kwargs)

    snapshots = n.snapshots if snapshots is None else snapshots

    if parallelized:
        f = lambda sn: func_dict[method](n, sn, **kwargs)
        res = xr.concat(parmap(f, snapshots, nprocs=nprocs))
    else:
        def f(sn):
            if sn.is_month_start & (sn.hour == 0):
                logger.info('Allocating for %s %s'%(sn.month_name(), sn.year))
            return func_dict[method](n, sn, **kwargs)
        res = xr.concat((f(sn) for sn in snapshots), dim=snapshots.rename('snapshot'))
    return res


