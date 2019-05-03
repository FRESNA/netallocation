import numpy as np
import pandas as pd

from .flow import flow_allocation as allocate_flow, network_flow


# Network Meta Data
####################################################################################################################
# km
def length(n):
    length_lines = pd.Series(n.lines.length.values,
                             pd.MultiIndex.from_tuples([('Line', idx) for idx in n.lines.length.index],
                             names=['component', 'branch_i']))
    length_links = pd.Series(n.links.length.values,
                             pd.MultiIndex.from_tuples([('Link', idx) for idx in n.links.length.index],
                             names=['component', 'branch_i']))
    length = length_lines.append(length_links)
    return length.rename('length')

# MW
def capacity(n):
    capacity_lines = pd.Series(n.lines.s_nom_opt.values,
                               pd.MultiIndex.from_tuples([('Line', idx) for idx in n.lines.s_nom_opt.index],
                               names=['component', 'branch_i']))
    capacity_links = pd.Series(n.links.p_nom_opt.values,
                               pd.MultiIndex.from_tuples([('Link', idx) for idx in n.links.p_nom_opt.index],
                               names=['component', 'branch_i']))
    capacity = capacity_lines.append(capacity_links)
    return capacity.rename('capacity')
####################################################################################################################


# Branch Cost Model
####################################################################################################################
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
####################################################################################################################


# Total Transmission Cost
####################################################################################################################
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
####################################################################################################################


# Pricing Strategies
####################################################################################################################
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
####################################################################################################################


# Main Function
####################################################################################################################
# €
def transmission_cost(n, snapshot,
                      allocation_method='Average participation',
                      pricing_strategy='MW-Mile', cost_model=None, normalisation=True,
                      allocation=None, cost_factors=None,
                      favor_counter_flows=True):

    if allocation is None: allocation = allocate_flow(n, snapshot, method=allocation_method)

    if cost_model is None: cost_model = pricing_strategy

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
        transmission_cost = transmission_cost * cost_normalisation(transmission_cost,
                                                                   total_transmission_cost(n, snapshot))

    return transmission_cost.rename('transmission cost')
####################################################################################################################
