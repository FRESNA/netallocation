import numpy as np
import pandas as pd
from pypsa.descriptors import get_switchable_as_dense as get_as_dense, get_extendable_i
from xarray import DataArray, Dataset, concat
import pypsa

from .flow import flow_allocation, network_flow
from .utils import reindex_by_bus_carrier, check_carriers, check_snapshots
from .convert import vip_to_p2p
from .breakdown import expand_by_source_type
from .grid import power_production, power_demand, Incidence

# ma = n.buses_t.marginal_price.loc[sns].T
# K = ntl.Incidence(n)
# transmission_cost = - K.T @ ma
# f = ntl.network_flow(n, sns).T
# tso_profit = transmission_cost * f
# revenue = n.loads_t.p.T.groupby(n.loads.bus).sum() * ma
# expenses = (n.generators_t.p * n.generators.marginal_cost).T\
#             .groupby(n.generators.bus).sum() + \
#             (n.storage_units_t.p * n.storage_units.marginal_cost).T\
#             .groupby(n.storage_units.bus).sum()
# revenue.sum() - expenses.sum() - tso_profit.sum()


def weight_with_generation_cost(n, ds, snapshots=None):
    """
    Allocate the production cost on the basis of a peer-to-peer allocation.

    Parameters
    ----------
    n : pypsa.Network
    ds : str/xarray.Dataset
        Allocation method, e.g. 'ap' or already calculated power allocation
        dataset, i.e. from ntl.allocate_flow.
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.

    Returns
    -------
    xr.DataArray
        Allocated generation cost.

    """
    if isinstance(ds, str):
        ds = flow_allocation(n, snapshots, ds)
    snapshots = check_snapshots(snapshots, n)
    check_carriers(n)
    ds = expand_by_source_type(ds, n)
    comps = ['Generator', 'StorageUnit', 'Store']
    gen = (reindex_by_bus_carrier(get_as_dense(n, c, 'marginal_cost', snapshots),
                                  c, n) for c in comps)
    prodcost_pu = concat(gen, dim='carrier')\
                  .rename(bus='source', carrier='source_carrier')
    return prodcost_pu * ds


def weight_with_branch_cost(n, ds, snapshots=None):
    """
    Allocate the branch cost on the basis of an allocation method.

    Parameters
    ----------
    n : pypsa.Network
    ds : str/xarray.Dataset
        Allocation method, e.g. 'ap' or already calculated power allocation
        dataset, i.e. from ntl.allocate_flow.
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
                              keys=['Link'], axis=1, names=['component', 'branch_i'])\
                        .loc[:, lambda df: (df > 0).any()]
    branchcost_pu = DataArray(branchcost_pu, dims='branch')
    return branchcost_pu * ds


def weight_with_carrier_attribute(n, ds, attr, snapshots=None):
    """
    Allocate an carrier attribute on the basis of a peer-to-peer allocation.

    Parameters
    ----------
    n : pypsa.Network
    ds : str/xarray.Dataset
        Allocation method, e.g. 'ap' or already calculated power allocation
        dataset, i.e. from ntl.allocate_flow.
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


def locational_market_price(n, snapshots=None, per_MW=False):
    """
    Get the locational market price (LMP) of 1 MWh in a solved network.

    If per_MW is set to True, the marginal price are mutliplied by
    n.snapshot_weightings, thus it reflects the price of 1 MW produced for
    elapsed time.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : subset of n.snapshots
        The default None results in taking all snapshots of n.
    per_MW : bool, optional
        See above. The default is False.

    Returns
    -------
    ma : xr.DataArray
        Marginal Price for dimensions {bus, snapshot}.

    """
    snapshots = check_snapshots(snapshots, n)
    ma = DataArray(n.buses_t.marginal_price.loc[snapshots], dims=['snapshot', 'bus'])
    if per_MW:
        ma *= DataArray(n.snapshot_weightings.loc[snapshots], dims='snapshot')
    return ma


def locational_market_price_diff(n, snapshots=None, per_MW=True):
    return Incidence(n) @ locational_market_price(n, snapshots, per_MW)


def congestion_revenue(n, snapshots=None, per_MW=True):
    return locational_market_price_diff(n, snapshots, per_MW) * \
           network_flow(n, snapshots)


def nodal_demand_cost(n, snapshots=None, per_MW=True):
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
    return power_demand(n, snapshots) * \
           locational_market_price(n, snapshots, per_MW)


def nodal_co2_cost(n, snapshots=None, co2_attr='co2_emissions',
                   co2_constr_name='CO2Limit'):
    c = 'Generator'
    price = n.global_constraints.mu[co2_constr_name]
    return price * reindex_by_bus_carrier(n.pnl(c).p / n.df(c).efficiency *
                                  n.df(c).carrier.map(n.carriers[co2_attr]), c, n)


def nodal_production_revenue(n, snapshots=None, per_MW=True):
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
    return power_production(n, snapshots) * \
           locational_market_price(n, snapshots, per_MW)


def objective_constant(n):
    nom_attr = pypsa.descriptors.nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = get_extendable_i(n, c)
        constant += n.df(c)[attr][ext_i] @ n.df(c).capital_cost[ext_i]
    return constant

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
