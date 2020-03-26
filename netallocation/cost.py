import numpy as np
import pandas as pd
from pypsa.descriptors import (get_switchable_as_dense as get_as_dense,
                               get_extendable_i, get_non_extendable_i)
from xarray import DataArray, Dataset, concat
import pypsa
import logging

from .flow import flow_allocation, network_flow
from .utils import (reindex_by_bus_carrier, check_carriers, check_snapshots,
                    get_branches_i, split_one_ports, split_branches,
                    snapshot_weightings)
from .convert import vip_to_p2p
from .breakdown import expand_by_source_type
from .grid import (Incidence, impedance, energy_production, energy_demand,
                   power_production)

logger = logging.getLogger(__name__)


def weight_with_generation_cost(ds, n, snapshots=None, dim='source',
                                add_co2_cost=False, co2_attr='co2_emissions',
                                co2_constr_name=None, **kwargs):
    """
    Allocate the production cost on the basis of a peer-to-peer allocation.

    Parameters
    ----------
    ds : str/xarray.Dataset
        Allocation method, e.g. 'ap' or already calculated power allocation
        dataset, i.e. from ntl.allocate_flow.
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
    if isinstance(ds, str):
        snapshots = check_snapshots(snapshots, n)
        ds = flow_allocation(n, snapshots, ds, **kwargs)
    else:
        snapshots = ds.snapshot.values
    check_carriers(n)
    ds = expand_by_source_type(ds, n, dim=dim).rename(source_carrier='carrier')
    comps = ['Generator', 'StorageUnit', 'Store']
    gen = (reindex_by_bus_carrier(
        get_as_dense(n, c, 'marginal_cost').loc[snapshots], c, n) for c in comps)
    prodcost_pu = concat(gen, dim='carrier')
    if add_co2_cost:
        ep = nodal_co2_price(n, snapshots, co2_attr, co2_constr_name)
        prodcost_pu = prodcost_pu + ep
    if 'name' in prodcost_pu.dims:
        prodcost_pu = prodcost_pu.rename(name='snapshot')
    prodcost_pu = prodcost_pu.rename(bus=dim)
    return prodcost_pu * ds


def weight_with_branch_cost(ds, n, snapshots=None):
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
    if isinstance(ds, str):
        ds = flow_allocation(n, snapshots, ds)
    snapshots = check_snapshots(snapshots, n)
    check_carriers(n)
    ds = expand_by_source_type(ds, n)

    branchcost_pu = pd.concat([get_as_dense(n, 'Link', 'marginal_cost', snapshots)],
                              keys=['Link'], axis=1,
                              names=['component', 'branch_i'])\
                        .loc[:, lambda df: (df > 0).any()]
    branchcost_pu = DataArray(branchcost_pu, dims='branch')
    return branchcost_pu * ds


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
        logger.warn(' The cost of cycle constraints cannot be calculated, as '
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
            logger.warn('No CO2 constraint found.')
            return np.array(0)
        else:
            co2_constr_name = co2_constr_name[0]
    elif co2_constr_name not in n.global_constraints.index:
        logger.warn(f'Constraint {co2_constr_name} not in n.global_constraints'
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


def nodal_production_revenue(n, snapshots=None, split=False):
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
        return split_one_ports(pr, n).sum('carrier')
    return energy_production(n, snapshots) * locational_market_price(n, snapshots)



def objective_constant(n):
    nom_attr = pypsa.descriptors.nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = get_extendable_i(n, c)
        constant += n.df(c)[attr][ext_i] @ n.df(c).capital_cost[ext_i]
    return constant


def allocate_cost(n, snapshots=None, method='ap', **kwargs):
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
    p2p = vip_to_p2p(ds, n).peer_to_peer
    return weight_with_generation_cost(p2p, n).sum('carrier')



# Total Production Revenue + Total Congetion Renvenue = Total Demand Cost
# (nodal_production_revenue(n).sum() + congestion_revenue(n).sum())/ nodal_demand_cost(n).sum() == 1

# Network Meta Data
# km
def length(n):
    return n.branches()['length'].rename_axis(['component', 'branch_i'])

# MW
def capacity(n):
    return n.branches().eval('s_nom_opt + p_nom_opt').rename_axis(['component', 'branch_i'])


# Branch Cost Model
# €/MWkm*[CostModelUnit]
def branch_cost_model(allocation, cost_model=None):
    if cost_model is None:
        raise NameError('No valid cost model for the given pricing strategy.')
    elif cost_model == 'MW-Mile':
        raise NotImplementedError(cost_model + ' yet not implemented!')
    elif cost_model == 'Capacity Pricing':
        raise NotImplementedError(cost_model + ' yet not implemented!')
    elif cost_model == 'Random Numbers':
        cost_factors = pd.Series(np.random.random(len(allocation.index)), allocation.index)
    elif cost_model == 'Unit Test':
        cost_factors = pd.Series(1.0, allocation.index)
    else:
        raise NameError('No valid cost model for the given pricing strategy.')
    return cost_factors.rename('branch cost model')

# €/MWkm*[CostModelUnit]
def normalised_branch_cost(n, allocation, cost_model=None, cost_factors=None):
    if cost_factors is None:
        cost_factors = branch_cost_model(allocation, cost_model=cost_model)

    normalised_branch_cost = cost_factors
    return normalised_branch_cost.rename('normalised branch cost')

# €/MW*[CostModelUnit]
def branch_cost(n, allocation, cost_model=None, cost_factors=None):
    branch_cost = normalised_branch_cost(n, allocation, cost_model=cost_model, cost_factors=cost_factors) * \
                  length(n)

    branch_cost = branch_cost.dropna()
    return branch_cost.rename('branch cost')


# Total Transmission Cost
def cost_normalisation(cost, total_cost):
    normalisation = total_cost / cost.sum()
    return normalisation

# €
def total_transmission_cost(n, snapshot):
    cap_lines = (n.lines['capital_cost'] * n.lines['s_nom_opt']).sum()

    cap_links = (n.links['capital_cost'] * n.links['p_nom_opt']).sum()
    mar_links = (n.links['marginal_cost'] * abs(n.links_t.p0.loc[snapshot])).sum().sum()

    total_cost = cap_lines + cap_links + mar_links
    return total_cost


# Pricing Strategies
def strategy_factor(n, allocation, pricing_strategy):
    if pricing_strategy is None:
        raise NameError('No valid pricing strategy.')
    elif pricing_strategy == 'MW-Mile':
        strategy_factor = pd.Series(1.0, allocation.index)
    elif pricing_strategy == 'Capacity Pricing':
        strategy_factor = (1.0/capacity(n)).replace(np.inf, np.nan).dropna()
    else:
        raise NameError('No valid pricing strategy.')
    return strategy_factor.rename('strategy factor')


# Main Function
# €
def transmission_cost(n, snapshot,
                      allocation_method='Average participation',
                      pricing_strategy='MW-Mile', cost_model=None, normalisation=True,
                      allocation=None, cost_factors=None,
                      favor_counter_flows=True):

    if allocation is None:
        allocation = flow_allocation(n, snapshot, method=allocation_method)
    if cost_model is None:
        cost_model = pricing_strategy
    if favor_counter_flows:
        sn = [snapshot] if isinstance(snapshot, pd.Timestamp) else snapshot
        f_dir = (network_flow(n, sn).applymap(np.sign)
                                          .unstack()
                                          .reorder_levels([2,0,1])
                                          .sort_index())
        allocation = (allocation/f_dir).dropna()\
                        .reorder_levels(allocation.index.names)
    else:
        allocation = abs(allocation)

    transmission_cost = (allocation * \
                         strategy_factor(n, allocation, pricing_strategy)) \
                        .reorder_levels(allocation.index.names) * \
                         branch_cost(n, allocation,
                                     cost_model=cost_model, cost_factors=cost_factors) \
                        .reorder_levels(allocation.index.names)

    if normalisation == True:
        normalized = cost_normalisation(transmission_cost,
                                        total_transmission_cost(n, snapshot))
        transmission_cost = transmission_cost * normalized

    return transmission_cost.rename('transmission cost')
