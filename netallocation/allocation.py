#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:14:49 2018

@author: fabian
"""

# This side-package is created for use as flow and cost allocation.

from .descriptors import Dict
import pandas as pd
import numpy as np
from collections import Iterable
import os
from numpy import sign
import logging
logger = logging.getLogger(__name__)

from .breakdown import expand_by_sink_type, expand_by_source_type
from .powergrid import (self_consumption, power_demand, power_production,
                        network_injection, network_flow, Incidence,
                        PTDF)
from .linalg import diag, inv, pinv


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


    f0 = pd.concat([n.pnl(c).p0.loc[snapshot] for c in branch_components],
                  keys=branch_components, sort=True) \
          .rename_axis(['component', 'branch_i'])
    f1 = pd.concat([n.pnl(c).p1.loc[snapshot] for c in branch_components],
                  keys=branch_components, sort=True) \
          .rename_axis(['component', 'branch_i'])

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

    Q = inv(lower(K_dir) @ diag(f_out) @ K.T + diag(p_in)) @ diag(p_in)
    R = inv(upper(K_dir) @ diag(f_in) @ K.T + diag(p_out)) @ diag(p_out)

    if not normalized and per_bus:
        Q = diag(p_out) @ Q
        R = diag(p_in) @ R
        if aggregated:
            # add self-consumption
            Q += diag(self_consumption(n, snapshot))
            R += diag(self_consumption(n, snapshot))

    q = (Q.rename_axis('in').rename_axis('source', axis=1)
         .replace(0, np.nan)
         .stack().swaplevel(0)#.sort_index()
         .rename('upstream'))#.pipe(set_cats, n))

    r = (R.rename_axis('out').rename_axis('sink', axis=1)
         .replace(0, np.nan)
         .stack()#.sort_index()
         .rename('downstream'))#.pipe(set_cats, n))


    if per_bus:
        T = (pd.concat([q,r], axis=0, keys=['upstream', 'downstream'],
                       names=['method', 'source', 'sink'])
             .rename('allocation'))
        if downstream is not None:
            T = T.downstream if downstream else T.upstream

    else:
        f = f_in if downstream else f_out

        f = (n.branches().loc[branch_components]
               .assign(flow=f)
               .rename_axis(['component', 'branch_i'])
               .set_index(['bus0', 'bus1'], append=True)['flow'])

        # absolute flow with directions
        f_dir = pd.concat(
                [f[f > 0].rename_axis(index={'bus0':'in', 'bus1': 'out'}),
                 f[f < 0].swaplevel()
                         .rename_axis(index={'bus0':'out', 'bus1': 'in'})])


        if normalized:
            f_dir = (f_dir.groupby(level=['component', 'branch_i'])
                     .transform(lambda ds: ds/ds.abs().sum()))

        T = (q * f_dir * r).dropna() \
            .droplevel(['in', 'out'])\
            .reorder_levels(['source', 'sink', 'component', 'branch_i'])\
            .rename('allocation')

    return pd.concat([T], keys=[snapshot], names=['snapshot'])


def marginal_participation(n, snapshot=None, q=0.5, normalized=False,
                           per_bus=False, branch_components=['Link', 'Line']):
    '''
    Allocate line flows according to linear sensitvities of nodal power
    injection given by the changes in the power transfer distribution
    factors (PTDF)[1-3]. As the method is based on the DC-approximation,
    it works on subnetworks only as link flows are not taken into account.
    Note that this method does not exclude counter flows.

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
    per_bus : Boolean, default True
        Whether to return allocation on buses. Allocate to lines
        if False.
    normalized : Boolean, default False
        Return the share of the source (sink) flow

    '''
    snapshot = n.snapshots[0] if snapshot is None else snapshot
    H = PTDF(n, branch_components=branch_components, snapshot=snapshot)
    K = Incidence(n, branch_components=branch_components)
    f = pd.concat([n.pnl(b).p0.loc[snapshot] for b in branch_components],
                   keys=branch_components)
    p = K @ f
    p_plus = p.clip(lower=0)
    p_minus = p.clip(upper=0)
#   unbalanced flow from positive injection:
    f_plus = H @ p_plus
    k_plus = (q * f - f_plus) / p_plus.sum()
    if normalized:
        Q = H.add(k_plus, axis=0).mul(p, axis=1).div(f, axis=0).round(10).T
    else:
        Q = H.add(k_plus, axis=0).mul(p, axis=1).round(10).T
    if per_bus:
        K = Incidence(n, branch_components=['Line'])
        Q = K @ Q.T
        Q = (Q.rename_axis('source').rename_axis('sink', axis=1)
             .stack().round(8)[lambda ds:ds != 0])
    else:
        Q = (Q.rename_axis('bus')
             .rename_axis(['component', 'branch_i'], axis=1)
             .unstack()
             .round(8)[lambda ds:ds != 0]
             .reorder_levels(['bus', 'component', 'branch_i'])
             .sort_index())
    return pd.concat([Q], keys=[snapshot], names=['snapshot'])


def virtual_injection_pattern(n, snapshot=None, normalized=False, per_bus=False,
                              downstream=True,
                              branch_components=['Line', 'Link']):
    """
    Sequentially calculate the load flow induced by individual
    power sources in the network ignoring other sources and scaling
    down sinks. The sum of the resulting flow of those virtual
    injection patters is the total network flow. This method matches
    the 'Marginal participation' method with q = 1.



    Parameters
    ----------
    network : pypsa.Network object with calculated flow data
    snapshot : str
        Specify snapshot which should be investigated. Must be
        in network.snapshots.
    per_bus : Boolean, default True
        Whether to return allocation on buses. Allocate to lines
        if False.
    normalized : Boolean, default False
        Return the share of the source (sink) flow

    """
    snapshot = n.snapshots[0] if snapshot is None else snapshot
    H = PTDF(n, branch_components=branch_components, snapshot=snapshot)
    f = pd.concat([n.pnl(b).p0.loc[snapshot] for b in branch_components],
                   keys=branch_components)
    K = Incidence(n, branch_components=branch_components)
    p = K @ f
    p_plus = p.clip(lower=0)
    p_minus = p.clip(upper=0)
    if downstream:
        indiag = diag(p_plus)
        offdiag = (p_minus.to_frame().dot(p_plus.to_frame().T)
                   .div(p_plus.sum()))
    else:
        indiag = diag(p_minus)
        offdiag = (p_plus.to_frame().dot(p_minus.to_frame().T)
                   .div(p_minus.sum()))
    vip = indiag + offdiag
    if per_bus:
        Q = (vip[indiag.sum() == 0].T
             .rename_axis('sink', axis=int(downstream))
             .rename_axis('source', axis=int(not downstream))
             .stack()[lambda ds:ds != 0]).abs()
#        switch to counter stream by Q.swaplevel(0).sort_index()
    else:
        Q = H.dot(vip).round(10).T
        if normalized:
            # normalized colorvectors
            Q /= f
        Q = (Q.rename_axis('bus') \
              .rename_axis(["component", 'branch_i'], axis=1)
              .unstack().round(8)
              .reorder_levels(['bus', 'component', 'branch_i'])
              .sort_index()
              [lambda ds: ds != 0])
    return pd.concat([Q], keys=[snapshot], names=['snapshot'])


def optimal_flow_shares(n, snapshot, method='min', downstream=True,
                        per_bus=False, **kwargs):
    """



    """
    from scipy.optimize import minimize
    H = PTDF(n)
    p = n.buses_t.p.loc[snapshot]
    p_plus = p.clip(lower=0)
    p_minus = p.clip(upper=0)
    pp = p.to_frame().dot(p.to_frame().T).div(p).fillna(0)
    if downstream:
        indiag = diag(p_plus)
        offdiag = (p_minus.to_frame().dot(p_plus.to_frame().T)
                   .div(p_plus.sum()))
        pp = pp.clip(upper=0).add(diag(pp)).mul(np.sign(p.clip(lower=0)))
        bounds = pd.concat([pp.stack(), pp.stack().clip(lower=0)], axis=1,
                           keys=['lb', 'ub'])

#                   .pipe(lambda df: df - np.diagflat(np.diag(df)))
    else:
        indiag = diag(p_minus)
        offdiag = (p_plus.to_frame().dot(p_minus.to_frame().T)
                   .div(p_minus.sum()))
        pp = pp.clip(lower=0).add(diag(pp)).mul(-np.sign(p.clip(upper=0)))
        bounds = pd.concat([pp.stack().clip(upper=0), pp.stack()], axis=1,
                           keys=['lb', 'ub'])
    x0 = (indiag + offdiag).stack()
    N = len(n.buses)
    if method == 'min':
        sign = 1
    elif method == 'max':
        sign = -1

    def minimization(df):
        return sign * (H.dot(df.reshape(N, N)).stack()**2).sum()

    constr = [
            #   nodal balancing
            {'type': 'eq', 'fun': lambda df: df.reshape(N, N).sum(0)},
            #    total injection of colors
            {'type': 'eq', 'fun': lambda df: df.reshape(N, N).sum(1)-p.values}
            ]

    #   sources-sinks-fixation
    res = minimize(minimization, x0, constraints=constr,
                   bounds=bounds, options={'maxiter': 1000}, tol=1e-5,
                   method='SLSQP')
    print(res)
    sol = pd.DataFrame(res.x.reshape(N, N), columns=n.buses.index,
                       index=n.buses.index).round(10)
    if per_bus:
        return (sol[indiag.sum()==0].T
                .rename_axis('sink', axis=int(downstream))
                .rename_axis('source', axis=int(not downstream))
                .stack()[lambda ds:ds != 0])
    else:
        return H.dot(sol).round(8)


def zbus_transmission(n, snapshot=None):
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
    slackbus = n.buses[(n.buses_t.v_ang == 0).all()].index[0]


    # linearised method, start from linearised admittance matrix
    y = 1.j * admittance(n, branch_components=['Line'])
    K = Incidence(n, branch_components=['Line'])

    Y = K @ diag(y) @ K.T  # Ybus matrix

    Z = pinv(Y)
    # set angle of slackbus to 0
    Z = Z.add(-Z.loc[slackbus])
    # DC-approximated S = P
    # S = n.buses_t.p.loc[[snapshot]].T
    V = n.buses.v_nom.to_frame(snapshot) * \
        (1 + 1.j * n.buses_t.v_ang.loc[[snapshot]].T).rename_axis('bus0')
    I = Y @ V
    assert all((I * V).apply(np.real)
                == network_injection(n, snapshots=snapshot).T)

    # -------------------------------------------------------------------------
    # nonlinear method start with full admittance matrix from pypsa
#    n.sub_networks.obj[0].calculate_Y()
    # Zbus matrix
#    Y = pd.DataFrame(n.sub_networks.obj[0].Y.todense(), buses, buses)
#    Z = pd.DataFrame(pinv(Y), buses, buses)
#    Z = Z.add(-Z.loc[slackbus])

    # -------------------------------------------------------------------------

    # y_sh = n.lines.set_index(['bus0', 'bus1']).eval('g_pu + 1.j * b_pu')

    A = (K * y).T @ Z # == diag(y) @ K.T @ Z == PTDF
         #+ Z.mul(y_sh, axis=0, level=0).set_axis(n.lines.index, inplace=False)
    A = A.applymap(np.real_if_close)
    branches = ['Line']
    f = pd.concat([n.pnl(b).p0.loc[snapshot] for b in branches], keys=branches)

    V_l_at = lambda bus: pd.concat([n.df(b)[bus].map(V[snapshot])
                                    / n.df(b)[bus].map(n.buses.v_nom ** 2)
                                    for b in branches], keys=branches)

    V_l = V_l_at('bus0').where(f > 0, V_l_at('bus1'))

    # q = PTDF(n) * p[snapshot]
    q = A.mul(V_l, axis=0)\
         .mul(I[snapshot]) \
         .applymap(np.real) \
         .stack() \
         .rename_axis(['component', 'branch_i', 'bus']) \
         .reorder_levels(['bus', 'component', 'branch_i'])\
         .sort_index()

    return pd.concat([q], keys=[snapshot], names=['snapshot'])


def with_and_without_transit(n, snapshots=None,
                             branch_components=['Line', 'Link']):
    regions = n.buses.country.unique()

    if not n.links.empty:
        from pypsa.allocation import admittance, Incidence, diag, pinv
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
                    key=None, parallelized=False, nprocs=None, to_hdf=False,
                    **kwargs):
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
            - 'Virtual injection pattern'/'vip'
                Sequentially calculate the load flow induced by
                individual power sources in the network ignoring other
                sources and scaling down sinks.
            - 'Least square color flows'/'mfs'


    Returns
    -------
    res : dict
        The returned dict consists of two values of which the first,
        'flow', represents the allocated flows within a mulitindexed
        pandas.Series with levels ['snapshot', 'bus', 'line']. The
        second object, 'cost', returns the corresponding cost derived
        from the flow allocation.
    """
#    raise error if there are no flows

    snapshots = n.snapshots if snapshots is None else snapshots
    snapshots = snapshots if isinstance(snapshots, Iterable) else [snapshots]
    if n.lines_t.p0.shape[0] == 0:
        raise ValueError('Flows are not given by the network, '
                         'please solve the network flows first')
    n.calculate_dependent_values()

    if method in ['Average participation', 'ap']:
        method_func = average_participation
    elif method in ['Marginal Participation', 'mp']:
        method_func = marginal_participation
    elif method in ['Virtual injection pattern', 'vip']:
        method_func = virtual_injection_pattern
    elif method in ['Minimal flow shares', 'mfs']:
        method_func = minimal_flow_shares
    elif method in ['Zbus transmission', 'zbus']:
        method_func = zbus_transmission
    else:
        raise(ValueError('Method not implemented, please choose one out of'
                         "['Average participation', "
                         "'Marginal participation',"
                         "'Virtual injection pattern',"
                         "'Least square color flows']"))

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

        gen = (pd.read_hdf(store, p_str(p)).pipe(set_cats, n) for p in periods)
        flow = pd.concat(gen)
        os.remove(store)

    elif parallelized:
        flow = pd.concat(parmap(f, snapshots, nprocs=nprocs))
    else:
        flow = pd.concat((f(sn) for sn in snapshots))
    return flow.rename('allocation')


def chord_diagram(allocation, lower_bound=0, groups=None, size=300,
                  save_path='/tmp/chord_diagram_pypsa'):
    """
    This function builds a chord diagram on the base of holoviews [1].
    It visualizes allocated peer-to-peer flows for all buses given in
    the data. As for compatibility with ipython shell the rendering of
    the image is passed to matplotlib however to the disfavour of
    interactivity. Note that the plot becomes only meaningful for networks
    with N > 5, because of sparse flows otherwise.


    [1] http://holoviews.org/reference/elements/bokeh/Chord.html

    Parameters
    ----------

    allocation : pandas.Series (MultiIndex)
        Series of power transmission between buses. The first index
        level ('source') represents the source of the flow, the second
        level ('sink') its sink.
    lower_bound : int, default is 0
        filter small power flows by a lower bound
    groups : pd.Series, default is None
        Specify the groups of your buses, which are then used for coloring.
        The series must contain values for all allocated buses.
    size : int, default is 300
        Set the size of the holoview figure
    save_path : str, default is '/tmp/chord_diagram_pypsa'
        set the saving path of your figure

    """

    import holoviews as hv
    hv.extension('matplotlib')
    from IPython.display import Image

    if len(allocation.index.levels) == 3:
        allocation = allocation[allocation.index.levels[0][0]]

    allocated_buses = allocation.index.levels[0] \
                      .append(allocation.index.levels[1]).unique()
    bus_map = pd.Series(range(len(allocated_buses)), index=allocated_buses)

    links = allocation.to_frame('value').reset_index()\
        .replace({'source': bus_map, 'sink': bus_map})\
        .sort_values('source').reset_index(drop=True) \
        [lambda df: df.value >= lower_bound]

    nodes = pd.DataFrame({'bus': bus_map.index})
    if groups is None:
        cindex = 'index'
        ecindex = 'source'
    else:
        groups = groups.rename(index=bus_map)
        nodes = nodes.assign(groups=groups)
        links = links.assign(groups=links['source']
                             .map(groups))
        cindex = 'groups'
        ecindex = 'groups'

    nodes = hv.Dataset(nodes, 'index')
    diagram = hv.Chord((links, nodes))
    diagram = diagram.opts(style={'cmap': 'Category20',
                                  'edge_cmap': 'Category20'},
                           plot={'label_index': 'bus',
                                 'color_index': cindex,
                                 'edge_color_index': ecindex
                                 })
    renderer = hv.renderer('matplotlib').instance(fig='png', holomap='gif',
                                                  size=size, dpi=300)
    renderer.save(diagram, 'example_I')
    return Image(filename='example_I.png', width=800, height=800)




