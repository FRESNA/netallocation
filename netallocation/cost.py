import numpy as np
import pandas as pd
from pypsa.descriptors import (get_switchable_as_dense as get_as_dense,
                               get_extendable_i, get_non_extendable_i,
                               nominal_attrs)
from xarray import DataArray, Dataset, concat
import pypsa
import logging

from .flow import flow_allocation, network_flow
from .utils import (reindex_by_bus_carrier, check_carriers, check_snapshots,
                    get_branches_i, split_one_ports, split_branches,
                    snapshot_weightings, split_one_ports)
from .convert import vip_to_p2p, virtual_patterns
from .breakdown import expand_by_source_type
from .grid import (Incidence, impedance, energy_production, energy_demand,
                   power_production)

logger = logging.getLogger(__name__)


def weight_with_operational_cost(ds, n, snapshots=None, dim='source',
                                add_co2_cost=False, co2_attr='co2_emissions',
                                co2_constr_name=None, **kwargs):
    """
    Allocate production costs on the basis of a peer-to-peer allocation.

    Parameters
    ----------
    ds : xarray.Dataset
        Calculated power allocation dataset, i.e. from ntl.allocate_flow.
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.
    dim : str
        Name of dimension to be expanded by carrier (must contain bus names).

    Returns
    -------
    xr.DataArray
        Allocated generation cost.

    """
    snapshots = check_snapshots(ds.snapshot, n)
    check_carriers(n)
    ds = expand_by_source_type(ds, n, dim=dim).rename(source_carrier='carrier')
    comps = ['Generator', 'StorageUnit', 'Store']
    _ = (reindex_by_bus_carrier(
         get_as_dense(n, c, 'marginal_cost').loc[snapshots], c, n) for c in comps)
    prodcost_pu = concat(_, dim='carrier')
    if add_co2_cost:
        ep = nodal_co2_price(n, snapshots, co2_attr, co2_constr_name)
        prodcost_pu = prodcost_pu + ep
    if 'name' in prodcost_pu.dims:
        prodcost_pu = prodcost_pu.rename(name='snapshot')
    prodcost_pu = prodcost_pu.rename(bus=dim)
    return prodcost_pu * ds


def weight_with_co2_cost(ds, n, snapshots=None, dim='source', co2_constr_name=None,
                         co2_attr='co2_emissions',  **kwargs):
    """
    Allocate production costs on the basis of a peer-to-peer allocation.

    Parameters
    ----------
    ds : xarray.Dataset
        Calculated power allocation dataset, i.e. from ntl.allocate_flow.
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.
    dim : str
        Name of dimension to be expanded by carrier (must contain bus names).

    Returns
    -------
    xr.DataArray
        Allocated generation cost.

    """
    snapshots = check_snapshots(ds.snapshot, n)
    check_carriers(n)
    ds = expand_by_source_type(ds, n, dim=dim).rename(source_carrier='carrier')
    ep = nodal_co2_price(n, snapshots, co2_attr, co2_constr_name).rename(bus=dim)
    return ep * ds



def weight_with_one_port_investment_cost(ds, n, dim='source', proportional=False,
                                         **kwargs):
    """
    Allocate investment costs on the basis of a peer-to-peer allocation.

    Parameters
    ----------
    ds : xarray.Dataset
        Calculated power allocation dataset, i.e. from ntl.allocate_flow.
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.
    dim : str
        Name of dimension to be expanded by carrier (must contain bus names).

    Returns
    -------
    xr.DataArray
        Allocated generation cost.

    """
    check_carriers(n)
    da = expand_by_source_type(ds, n).rename(source_carrier='carrier').peer_to_peer

    comps = ['Generator']
    attr = nominal_attrs
    nom_opt = concat([reindex_by_bus_carrier(n.df(c)[attr[c] + "_opt"], c, n)
                      for c in comps], dim='carrier')
    cap_cost = concat((reindex_by_bus_carrier(n.df(c).capital_cost, c, n)
               for c in comps), dim='carrier')
    investment_cost = (nom_opt * cap_cost).rename(bus=dim)

    if not proportional:
        # 1. Empirical approach
        # nom_bound_pu = concat([reindex_by_bus_carrier(
        #                     get_as_dense(n, c, 'p_max_pu'), c, n)
        #                   for c in comps], dim='carrier').rename(name='snapshot')
        # nom_bound = nom_bound_pu * nom_opt

        # at_bound = nom_bound.round(3) \
        #            == power_production(n, per_carrier=True).round(3)
        # at_bound = at_bound.rename(bus=dim)\
        #                    .reindex_like(da.drop('sink'), fill_value=False)
        # da = da.where(at_bound, 0)

        # 2. Empirical approach
        # c = 'Generator'
        # at_bound = (get_as_dense(n, c, 'p_max_pu') * n.df(c).p_nom_opt /
        #             n.pnl(c).p.reindex(columns=n.generators.index) <= 1.001)
        # at_bound = (reindex_by_bus_carrier(at_bound, c, n)
        #             .rename(name='snapshot', bus=dim)
        #             .reindex_like(da.drop('sink'), fill_value=False))
        # da = da.where(at_bound, 0)

        # 4. Scale with binding of generator capacity
        c = 'Generator'
        mu_upper = n.pnl(c).mu_upper
        scaling = (reindex_by_bus_carrier(mu_upper/mu_upper.sum(), c, n)
                   .rename(bus=dim)
                   .reindex_like(da.drop('sink'), fill_value=0))
        da = da * scaling

    normed = (da / da.sum(['snapshot', 'sink'])).fillna(0)
    return investment_cost * normed



def weight_with_branch_operational_cost(ds, n, snapshots=None):
    """
    Allocate the branch cost on the basis of an allocation method.

    Parameters
    ----------
    ds : str/xarray.Dataset
        Allocation method, e.g. 'ap' or already calculated power allocation
        dataset, i.e. from ntl.allocate_flow.
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.

    Returns
    -------
    xr.DataArray
        Allocated branch cost.

    """
    snapshots = check_snapshots(snapshots, n)
    check_carriers(n)
    ds = expand_by_source_type(ds, n)

    branchcost_pu = pd.concat([get_as_dense(n, 'Link', 'marginal_cost', snapshots)],
                              keys=['Link'], axis=1,
                              names=['component', 'branch_i'])\
                        .loc[:, lambda df: (df > 0).any()]
    branchcost_pu = DataArray(branchcost_pu, dims='branch')
    return branchcost_pu * ds



def weight_with_branch_investment_cost(ds, n, snapshots=None):
    """
    Allocate the branch cost on the basis of an allocation method.

    Parameters
    ----------
    ds : str/xarray.Dataset
        Allocation method, e.g. 'ap' or already calculated power allocation
        dataset, i.e. from ntl.allocate_flow.
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.

    Returns
    -------
    xr.DataArray
        Allocated branch cost.

    """
    snapshots = check_snapshots(ds.snapshot, n)
    check_carriers(n)
    da = virtual_patterns(ds, n).virtual_flow_pattern
    names=['component', 'branch_i']

    nom_attr = pd.Series(nominal_attrs)[np.unique(da.component)] + '_opt'
    investment_cost = pd.concat({c: n.df(c).eval(f'capital_cost * {attr}')
                          for c, attr in nom_attr.items()}, names=names)
    investment_cost = DataArray(investment_cost, dims='branch')

    scaling = pd.concat({c: n.pnl(c).mu_lower.loc[snapshots]
                         + n.pnl(c).mu_upper.loc[snapshots]
                       for c in nom_attr.index}, axis=1, names=names)
    scaling = DataArray(scaling/scaling.sum(), dims=['snapshot', 'branch'])

    da = da * scaling

    normed = (da / da.sum(['snapshot', 'bus'])).fillna(0)
    return investment_cost * normed



def weight_with_carrier_attribute(ds, n, attr, snapshots=None):
    """
    Allocate an carrier attribute on the basis of a peer-to-peer allocation.

    Parameters
    ----------
    ds : str/xarray.Dataset
        Allocation method, e.g. 'ap' or already calculated power allocation
        dataset, i.e. from ntl.allocate_flow.
    n : pypsa.Network
    attr : str/pd.Series/pd.DataFrame
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.

    Returns
    -------
    xr.DataArray

    """
    snapshots = check_snapshots(snapshots, n)
    if isinstance(ds, str):
        ds = flow_allocation(n, snapshots, ds)
    check_carriers(n)
    ds = expand_by_source_type(ds, n)
    return DataArray(n.carriers[attr], dims='source_carrier') * ds


def locational_market_price(n, snapshots=None):
    """
    Get the locational market price (LMP) of 1 MWh in a solved network.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.

    Returns
    -------
    ma : xr.DataArray
        Marginal Price for dimensions {bus, snapshot}.

    """
    snapshots = check_snapshots(snapshots, n)
    ma = n.buses_t.marginal_price.loc[snapshots]
    if isinstance(ma, pd.Series):
        return DataArray(ma, dims=['bus']).assign_coords(snapshot=snapshots)
    else:
        return DataArray(ma, dims=['snapshot', 'bus'])



def locational_market_price_diff(n, snapshots=None):
    return Incidence(n) @ locational_market_price(n, snapshots)


def cycle_constraint_cost(n, snapshots=None):
    """
    Calculate the cost per branch and snapshot for the cycle constraint

    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.

    Returns
    -------
    xr.DataArray
        Cycle cost for passive branches, dimension {snapshot, branch}.

    """
    # Interesting fact: The cycle constraint cost are not weighted with
    # snapshot_weightings if they or not 1.
    C = []; i = 0
    for sub in n.sub_networks.obj:
        coords={'branch': sub.branches().index.rename(('component', 'branch_i')),
                'cycle': range(i, i+sub.C.shape[1])}
        i += sub.C.shape[1]
        C.append(DataArray(sub.C.todense(), coords, ['branch', 'cycle']))
    C = concat(C, dim='cycle').fillna(0)
    snapshots = check_snapshots(snapshots, n)
    comps = n.passive_branch_components
    f = network_flow(n, snapshots, comps)
    z = impedance(n, comps)
    sp = n.sub_networks_t.mu_kirchhoff_voltage_law.loc[snapshots]
    shadowprice = DataArray(sp, dims=['snapshot', 'cycle']) * 1e5
    return (C * z * f * shadowprice).sum('cycle')


def congestion_revenue(n, snapshots=None, split=False):
    """
    Calculate the congestion revenue (CR) per brnach and snapshot.

    The CR includes all costs of the transmission system. The sum over all
    snapshots of this is equal to the capital investments per branch.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.
    split : bool, optional
        If True, two CRs are returned, one indicating the renvenue for
        extendable branches, one for non-extendable branches.

    Example
    -------

    >>>>

    Returns
    -------
    xr.DataArray
        Congestion Revenue, dimension {snapshot, branch}.

    """
    cr = locational_market_price_diff(n, snapshots) * \
         network_flow(n, snapshots) * snapshot_weightings(n, snapshots)
    if 'mu_kirchhoff_voltage_law' in n.sub_networks_t:
        cr += cycle_constraint_cost(n, snapshots).reindex_like(cr, fill_value=0)
    else:
        logger.warning(' The cost of cycle constraints cannot be calculated, as '
                    'the shadowprices for those are missing. Please solve the '
                    'network with `keep_shadowprices=True` for including them.')
    return split_branches(cr, n) if split else cr


def nodal_demand_cost(n, snapshots=None):
    """
    Calculate the nodal demand cost per bus and snapshot.

    This is calculated by the product of power demand times the marginal price.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.

    """
    snapshots = check_snapshots(snapshots, n)
    return energy_demand(n, snapshots) * locational_market_price(n, snapshots)


def nodal_co2_price(n, snapshots=None, co2_attr='co2_emissions',
                    co2_constr_name=None):
    """
    Get the CO2 price per MWh_el.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.
    co2_attr : str, optional
        Name of the co2 emission attribut in n.carriers. The default is
        'co2_emissions'.
    co2_constr_name : str, optional
        Name of the CO2 limit constraint in n.global_constraint.
        The default is None will lead to searching for constraints which
        contain the strings "CO2" and "Limit".

    """
    c = 'Generator'
    if co2_constr_name is None:
        con_i = n.global_constraints.index
        co2_constr_name = con_i[con_i.str.contains('CO2', case=False) &
                                con_i.str.contains('Limit', case=False)]
        if co2_constr_name.empty:
            logger.warning('No CO2 constraint found.')
            return np.array(0)
        else:
            co2_constr_name = co2_constr_name[0]
    elif co2_constr_name not in n.global_constraints.index:
        logger.warning(f'Constraint {co2_constr_name} not in n.global_constraints'
                    ', setting CO₂ constraint cost to zero.')
        return np.array(0)
    price = n.global_constraints.mu[co2_constr_name]
    eff_emission = n.df(c).carrier.map(n.carriers[co2_attr]) / n.df(c).efficiency
    return price * reindex_by_bus_carrier(eff_emission, c, n)


def nodal_co2_cost(n, snapshots=None, co2_attr='co2_emissions',
                   co2_constr_name=None, split=False):
    """
    Calculate the total system cost caused by the CO2 constraint.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.
    co2_attr : str, optional
        Name of the co2 emission attribut in n.carriers. The default is
        'co2_emissions'.
    co2_constr_name : str, optional
        Name of the CO2 limit constraint in n.global_constraint.
        The default is None will lead to searching for constraints which
        contain the strings "CO2" and "Limit".

    """
    price_per_gen = nodal_co2_price(n, snapshots, co2_attr, co2_constr_name)
    cost = (energy_production(n, snapshots, per_carrier=True) * price_per_gen)
    if split:
        cost = split_one_ports(cost, n)
    return cost.sum('carrier')


def nodal_production_revenue(n, snapshots=None, split=False, per_carrier=False):
    """
    Calculate the nodal production revenue per bus and snapshot.

    This is calculated by the product of power production times the marginal
    price.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.

    """
    snapshots = check_snapshots(snapshots, n)
    if split:
        pr = energy_production(n, snapshots, per_carrier=True) * \
             locational_market_price(n, snapshots)
        pr = split_one_ports(pr, n)
        return pr if per_carrier else pr.sum('carrier')
    return energy_production(n, snapshots, per_carrier=per_carrier) * \
           locational_market_price(n, snapshots)



def objective_constant(n):
    nom_attr = nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = get_extendable_i(n, c)
        constant += n.df(c)[attr][ext_i] @ n.df(c).capital_cost[ext_i]
    return constant


def allocate_cost(n, snapshots=None, method='ap', add_investment_cost=True,
                  add_co2_cost=True, **kwargs):
    """
    Allocate production cost based on an allocation method.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots
    method : str/xarray.Dataset
        Method on which the cost allocation is based. Must be an available
        method for ``netallocation.allocate_flow``. Alternatively to a string,
        an calculated allocation xarray.Dataset can be passed.

    Returns
    -------
    xarray.DataArray
        Peer-to-peer cost allocation.

    """
    ds = flow_allocation(n, snapshots, method, **kwargs)
    ds = vip_to_p2p(ds, n)
    p2p = ds.peer_to_peer
    res = weight_with_operational_cost(p2p, n, add_co2_cost=add_co2_cost,
                                       **kwargs).sum('carrier')
    if add_investment_cost:
        res += weight_with_one_port_investment_cost(ds, n).sum('carrier')
    return res


