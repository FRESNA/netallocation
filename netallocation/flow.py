#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:14:49 2018

@author: fabian
"""

# This side-package is created for use as flow and cost allocation.

from .linalg import dot
from .grid import (self_consumption, power_demand, power_production,
                        network_injection, network_flow, Incidence,
                        PTDF, CISF, voltage, Ybus)
from .linalg import diag, inv, dedup_axis
from .utils import (upper, lower, as_sparse, check_branch_comps,
                    check_passive_branch_comps, check_snapshots)
import pandas as pd
import xarray as xr
from xarray import Dataset, DataArray
from numpy import real, conj, sign
import logging
from progressbar import ProgressBar

logger = logging.getLogger(__name__)

def average_participation(n, snapshot, dims='all',
                    branch_components=None, aggregated=True, downstream=True,
                    include_self_consumption=True, sparse=False, round=None):
    """
    Perform a Flow Tracing allocation.

    Allocate the network flow in according to the method 'Average
    participation' or 'Flow tracing' firstly presented in [1,2].
    The algorithm itself is derived from [3]. The general idea is to
    follow active power flow from source to sink (or sink to source)
    using the principle of proportional sharing and calculate the
    partial flows on each line, or to each bus where the power goes
    to (or comes from).

    This method provides two general options:
        Downstream:
            The flow of each nodal power injection is traced through
            the network and decomposed the to set of lines/buses
            on which is flows on/to.
        Upstream:
            The flow of each nodal power demand is traced
            (in reverse direction) through the network and decomposed
            to the set of lines/buses where it comes from.

    Note that only one snapshot can be calculated at a time, use
    `flow_allocation` to calculate multiple snapshots.


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
    n : pypsa.Network
        Network object with valid flow data.
    snapshot : str, pd.Timestamp
        Specify snapshot which should be investigated. Must be
        in network.snapshots.
    branch_components : list
        Components for which the allocation should be calculated.
        The default is None, which results in n.branch_components.
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
    sparse: boolean, default False
        Whether to compute the allocation with sparse arrays, this can save
        time for large networks
    round: float, default None
        Round the resulting allocation to a given precision in decimal digits.

    """
    dims = ['source', 'branch', 'sink'] if dims == 'all' else dims

    f0 = network_flow(n, snapshot, branch_components)
    f1 = network_flow(n, snapshot, branch_components, ingoing=False)
    f_in = f0.where(f0 > 0, - f1)
    f_out = f0.where(f0 < 0,  - f1)
    p = network_injection(n, snapshot, branch_components)

    if aggregated:
        # nodal inflow and nodal outflow
        p_in = upper(p).rename(bus='source')
        p_out = - lower(p).rename(bus='sink')
    else:
        p_in = power_production(n, [snapshot]).loc[snapshot].rename(bus='source')
        p_out = power_demand(n, [snapshot]).loc[snapshot].rename(bus='sink')

    K = Incidence(n, branch_components, sparse=sparse)
    K_dir = K * sign(f_in)

    newdims, newdims_r = ('source', 'sink'), ('sink', 'source')
    P_in = diag(p_in, newdims, sparse=sparse)
    P_out = diag(p_out, newdims, sparse=sparse)
    J = inv(dedup_axis(dot(lower(K_dir) * f_out, K.T), newdims) + P_in, True)
    Q = J * p_in
    J = inv(dedup_axis(dot(upper(K_dir) * f_in, K.T), newdims_r) + P_out, True)
    R = J * p_out

    if downstream:
        A, kind = Q * p_out, 'downstream'
    else:
        A, kind = R * p_in, 'upstream'

    if aggregated and include_self_consumption:
        selfcon = self_consumption(n, snapshot)
        if sparse:
            A += as_sparse(diag(selfcon, ('source', 'sink')))
        else:
            A += diag(selfcon, ('source', 'sink'))

    A = A.round(round) if round is not None else A
    res = A.to_dataset(name='peer_to_peer')\
           .assign_attrs(method='Average Participation')

    if 'branch' in dims:
        f = f_in if downstream else f_out
        T = dot(f * upper(K_dir.T), Q.fillna(0)) * dot(lower(K_dir.T), -R.fillna(0))
        T = T.assign_coords(snapshot=snapshot).assign_attrs(kind=kind)
        T = T.round(round) if round is not None else T
        res = res.assign({'peer_on_branch_to_peer': T})
    if round is not None:
        res = res.round(round).assign_attrs(res.attrs)
    return res





def marginal_participation(n, snapshot=None, q=0.5, branch_components=None,
                           sparse=False, round=None):
    """
    Perform a Marginal Participation allocation.

    Allocate line flows according to linear sensitvities of nodal power
    injection given by the changes in the power transfer distribution
    factors (PTDF)[1-3]. As the method is based on the DC-approximation,
    it works on subnetworks only as link flows are not taken into account.
    This method does not exclude counter flows.
    Note that only one snapshot can be calculated at a time, use
    `flow_allocation` to calculate multiple snapshots.


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
    n : pypsa.Network
        Network object with valid flow data.
    snapshot : str, pd.Timestamp
        Specify snapshot which should be investigated. Must be
        in network.snapshots.
    branch_components : list
        Components for which the allocation should be calculated.
        The default is None, which results in n.branch_components.
    q : float, default 0.5
        split between net producers and net consumers.
        If q is zero, only the impact of net load is taken into
        account. If q is one, only net generators are taken
        into account.
    round: float, default None
        Round the resulting allocation to a given precision in decimal digits.

    """
    snapshot = n.snapshots[0] if snapshot is None else snapshot
    H = PTDF(n, branch_components=branch_components, snapshot=snapshot)
    K = Incidence(n, branch_components=branch_components)
    f = network_flow(n, [snapshot], branch_components)
    p = K @ f
    p_plus = upper(p)
    p_minus = lower(p)
    new_dims = ('bus', 'injection_pattern')
    P = diag(p.loc[:, snapshot], new_dims)
    s = 0.5 - abs(q - 0.5)
    gamma = float(p_plus.sum())
    A = dedup_axis(dot(p_minus, p_plus.T) / gamma, new_dims)
    B = dedup_axis(dot(p_plus, p_minus.T) / gamma, new_dims)
    C = dedup_axis(dot(p_minus, p_minus.T) / gamma, new_dims)
    D = dedup_axis(dot(p_plus, p_plus.T) / gamma, new_dims)
    P = (q * (upper(P) + A) + (1 - q) * (lower(P) - B)
          + s * (P + C - D)).assign_coords(snapshot = snapshot)
    F = (H @ P).rename(injection_pattern='bus')
    res = Dataset({'virtual_injection_pattern': P, 'virtual_flow_pattern': F},
                  attrs={'method': 'Marginal Participation'})
    if round is not None:
        res = res.round(round).assign_attrs(res.attrs)
    return as_sparse(res) if sparse else res



def equivalent_bilateral_exchanges(n, snapshot=None, branch_components=None,
                                   q=0.5, sparse=False, round=None):
    """
    Perform a Equivalent Bilateral Exchanges allocation.

    Calculate the load flow induced by individual
    power sources in the network ignoring other sources and scaling
    down sinks. The sum of the resulting flow of those virtual
    injection patters is the total network flow. This method matches
    the 'Marginal participation' method for q = 1. Return either Virtual
    Injection Patterns if vip is set to True, or Virtual Flow Patterns.
    Note that only one snapshot can be calculated at a time, use
    `flow_allocation` to calculate multiple snapshots.

    Parameters
    ----------
    n : pypsa.Network
        Network object with valid flow data.
    snapshot : str, pd.Timestamp
        Specify snapshot which should be investigated. Must be
        in network.snapshots.
    branch_components : list
        Components for which the allocation should be calculated.
        The default is None, which results in n.branch_components.
    round: float, default None
        Round the resulting allocation to a given precision in decimal digits.

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
    new_dims = ('bus', 'injection_pattern')
    A = dedup_axis(dot(p_minus, p_plus.T) / float(p_pl.sum()), new_dims)
    B = dedup_axis(dot(p_plus, p_minus.T) / float(p_pl.sum()), new_dims)
    P = q * (A + diag(p_pl, new_dims)) + (q - 1) * (B - diag(p_min, new_dims))
    P = P.assign_coords(snapshot = snapshot)
    F = (H @ P).rename(injection_pattern='bus')
    res = Dataset({'virtual_injection_pattern': P, 'virtual_flow_pattern': F},
                  attrs={'method': 'Eqivalent Bilateral Exchanges'})
    if round is not None:
        res = res.round(round).assign_attrs(res.attrs)
    return as_sparse(res) if sparse else res


def zbus_transmission(n, snapshot=None, linear=True, downstream=None,
                      branch_components=None):
    r"""
    Perform a Zbus Transmission allocation.

    This allocation builds up on the method presented in [1]. However, we
    provide for non-linear power flow an additional DC-approximated
    modification, neglecting the series resistance r for lines.


    [1] A. J. Conejo, J. Contreras, D. A. Lima, and A. Padilha-Feltrin,
        “$Z_{\rm bus}$ Transmission Network Cost Allocation,” IEEE Transactions
        on Power Systems, vol. 22, no. 1, pp. 342–349, Feb. 2007.

    Parameters
    ----------
    n : pypsa.Network
        Network object with valid flow data.
    snapshot : str, pd.Timestamp, list, pd.Index
        Specify snapshot(s) for which the allocation should be performed.
        Must be a suset of n.snapshots.
    branch_components : list
        Components for which the allocation should be calculated.
        The default is None, which results in n.branch_components.

    """
    n.calculate_dependent_values()
    branch_components = check_passive_branch_comps(branch_components, n)
    snapshot = check_snapshots(snapshot, n)
    assert 'Link' not in branch_components, ('Component "Link" cannot be '
                'considered in Zbus flow allocation.')

    K = Incidence(n, branch_components=branch_components)
    Y = Ybus(n, branch_components, linear=linear)  # Ybus matrix
    v = voltage(n, snapshot, linear=linear)
    H = PTDF(n, branch_components) if linear else CISF(n, branch_components)
    i = dot(Y, v)
    f = network_flow(n, snapshot, branch_components)
    if downstream is None:
        v_ = abs(K) @ v / 2
    elif downstream:
        v_ = upper(K * sign(f)) @ v
    else:
        v_ = -lower(K * sign(f)) @ v

    if linear:
        # i == network_injection(n, snapshot, branch_components=branch_components)
        vif = H * i # which is the same as mp with q=0.5
    else:
        # real(conj(i) * v) == n.buses_t.p.loc[snapshot].T
        vif = real( v_ * conj(H) * conj(i))
    vif = vif.transpose(..., 'branch', 'bus')
    vip = K.dot(vif.rename(bus='injection_pattern'), 'branch')\
           .transpose(..., 'bus', 'injection_pattern')
    ds = Dataset({'virtual_flow_pattern': vif,
                  'virtual_injection_pattern': vip},
                  attrs={'method': 'Zbus flow allocation'})
    if isinstance(snapshot, pd.Index):
        return ds
    return ds.assign_coords(snapshot=snapshot)


def with_and_without_transit(n, snapshots=None, branch_components=None):
    """
    Compute the with-and-without flows and losses.

    This function only works with the linear power so far and calculated the
    loss which *would* take place accoring to

    f²⋅r

    which is the loss for directed currents. If links are included their
    efficiency determines the loss.

    Parameters
    ----------
    n : pypsa.Network
        Network object with valid flow data.
    snapshots : pd.Index or list
        Snapshots for which the flows and losses are calculated. Thye must be
        a subset of n.snapshots. The default is None, which results
        in n.snapshots.
    branch_components : list
        Components for which the allocation should be calculated.
        The default is None, which results in n.passive_branch_components.

    Returns
    -------
    xarray.Dataset
        Resulting loss allocation of dimension {branch, country, snapshot} with
        variables [flow_with, loss_with, flow_without, loss_without].

    """
    branch_components = check_passive_branch_comps(branch_components, n)
    snapshots = check_snapshots(snapshots, n)
    regions = pd.Index(n.buses.country.unique(), name='country')
    branches = n.branches().loc[branch_components]
    f = network_flow(n, snapshots, branch_components)

    def regional_with_and_withtout_flow(region):
        in_region_buses = n.buses.query('country == @region').index
        region_branches = branches.query('bus0 in @in_region_buses '
                                         'or bus1 in @in_region_buses')
        buses_i = (pd.Index(region_branches.bus0.unique()) |
                   pd.Index(region_branches.bus1.unique()) |
                   in_region_buses)
        vicinity_buses = buses_i.difference(in_region_buses)
        branches_i = region_branches.index

        K = Incidence(n, branch_components).loc[buses_i]
        #create regional injection pattern with nodal injection at the border
        #accounting for the cross border flow
        p = (K @ f)
        # p.loc[in_region_buses] ==
        #     network_injection(n, snapshots).loc[snapshots, in_region_buses].T

        #modified injection pattern without transition
        im = upper(p.loc[vicinity_buses])
        ex = lower(p.loc[vicinity_buses])

        largerImport_b = im.sum('bus') > - ex.sum('bus')
        scaleImport = (im.sum('bus') + ex.sum('bus')) / im.sum('bus')
        scaleExport = (im.sum('bus') + ex.sum('bus')) / ex.sum('bus')
        netImOrEx = (im * scaleImport).where(largerImport_b, (ex * scaleExport))
        p_wo = xr.concat([p.loc[in_region_buses], netImOrEx], dim='bus')\
                 .reindex(bus=buses_i).fillna(0)

        if 'Link' in branch_components:
            H = xr.concat((PTDF(n, branch_components, snapshot=sn)
                           for sn in snapshots), dim='snapshot')\
                  .sel(branch=branches_i)
            # f == H @ p
        else:
            H = PTDF(n, branch_components).sel(bus=branches_i)
        f_wo = H.reindex(bus=buses_i).dot(p_wo, 'bus')

        res = Dataset({'flow_with_transit': f.sel(branch=branches_i),
                        'flow_without_transit': f_wo})\
                    .assign_coords(country=region)
        return res.assign(transit_flow = res.flow_with_transit -
                          res.flow_without_transit)

    progress = ProgressBar()
    flows = xr.concat((regional_with_and_withtout_flow(r) for r in progress(regions)),
                      dim='country')
    comps = flows.get_index('branch').unique('component')
    loss = xr.concat((flows.sel(component=c)**2 * DataArray(n.df(c).r_pu, dims='branch_i')
           if c in n.passive_branch_components else
           flows.sel(component=c) * DataArray(n.df(c).efficiency, dims='branch_i')
           for c in comps), dim=comps).stack(branch=['component', 'branch_i'])\
           .rename_vars(flow_with_transit = 'loss_with_transit',
                        flow_without_transit = 'loss_without_transit',
                        transit_flow = 'transit_flow_loss')
    return flows.merge(loss).assign_attrs(method='With-and-Without-Transit').fillna(0)


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

    snapshots = check_snapshots(snapshots, n)
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

_func_dict = {'Average participation': average_participation,
             'ap': average_participation,
             'Marginal participation': marginal_participation,
             'mp': marginal_participation,
             'Equivalent bilateral exchanges': equivalent_bilateral_exchanges,
             'ebe': equivalent_bilateral_exchanges,
             'Zbus transmission': zbus_transmission,
             'zbus': zbus_transmission}
_non_sequential_funcs = [zbus_transmission, with_and_without_transit]

def flow_allocation(n, snapshots=None, method='Average participation',
                    round_floats=8, **kwargs):
    """
    Allocate or decompose the network flow with different methods.

    Available methods are 'Average participation' ('ap'), 'Marginal
    participation' ('mp'), 'Virtual injection pattern' ('vip'),
    'Zbus transmission' ('zbus').



    Parameters
    ----------
    n : pypsa.Network
        Network object with valid flow data.
    snapshots : string or pandas.DatetimeIndex
        (Subset of) snapshots of the network. If None (dafault) all snapshots
        are taken.
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
            - 'Zbus transmission'/'zbus'

    Returns
    -------
    res : xr.Dataset
        Dataset with allocations depending on the method.
    """
    snapshots = check_snapshots(snapshots, n)
    n.calculate_dependent_values()
    if all(c.pnl.p0.empty for c in n.iterate_components(n.branch_components)):
        raise ValueError('Flows are not given by the network, '
                         'please solve the network flows first')

    if method not in _func_dict.keys():
        raise(ValueError('Method not implemented, please choose one out of'
                         f'{list(_func_dict.keys())}'))

    is_nonsequetial_func = _func_dict[method] in _non_sequential_funcs
    if isinstance(snapshots, (str, pd.Timestamp)) or is_nonsequetial_func:
        return _func_dict[method](n, snapshots, **kwargs)

    pbar = ProgressBar(prefix='Calculate allocations')
    func = lambda sn: _func_dict[method](n, sn, **kwargs)
    res = xr.concat((func(sn) for sn in pbar(snapshots)),
                    dim=snapshots.rename('snapshot'))
    return res


