import numpy as np
import pandas as pd
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from xarray import DataArray, Dataset

from .flow import flow_allocation, network_flow
from .utils import group_per_bus_carrier
from .convert import vip_to_p2p
from .breakdown import expand_by_source_type
from .grid import power_production, power_demand

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


def p2p_allocated_sink_costs(n, ds, snapshots=None):
    if snapshots is None:
        snapshots = n.snapshots
    if isinstance(ds, str):
        ds = flow_allocation(n, snapshots, ds)
    ds = expand_by_source_type(ds, n)
    if 'carrier' not in n.stores:
        n.stores['carrier'] = n.stores.bus.map(n.buses.carrier)
    comps = ['Generator', 'StorageUnit', 'Store']
    prodcost_pu = DataArray(pd.concat(
            (group_per_bus_carrier(get_as_dense(n, c, 'marginal_cost', snapshots), c, n)
             for c in comps), axis=1), dims=['snapshot', 'marginal_cost'])\
            .unstack('marginal_cost', fill_value=0)\
            .rename(bus='source', carrier='source_carrier')

    branchcost_pu = pd.concat([get_as_dense(n, 'Link', 'marginal_cost', snapshots)],
                              keys=['Link'], axis=1, names=['component', 'branch_i'])\
                        .loc[:, lambda df: (df > 0).any()]
    branchcost_pu = DataArray(branchcost_pu).unstack('dim_1')

    cost_per_sink_alloc = (prodcost_pu * ds).sum(['source', 'source_carrier']).peer_to_peer

    marg_price = DataArray(n.buses_t.marginal_price, dims=['snapshot', 'sink'])
    cost_per_sink_opt = power_demand(n, snapshots).rename(bus='sink') * marg_price

    return Dataset({'from_allocation': cost_per_sink_alloc,
                    'from_shadow_price': cost_per_sink_opt})



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
