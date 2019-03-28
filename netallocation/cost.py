import numpy as np
import pandas as pd

from . import allocate_flow


# Network Meta Data
####################################################################################################################
# km
def length(network):
    length_lines = pd.Series(network.lines.length.values,
                             pd.MultiIndex.from_tuples([('Line', idx) for idx in network.lines.length.index],
                             names=['component', 'branch_i']))
    length_links = pd.Series(network.links.length.values,
                             pd.MultiIndex.from_tuples([('Link', idx) for idx in network.links.length.index],
                             names=['component', 'branch_i']))
    length = length_lines.append(length_links)
    return length.rename('length')

# MW
def capacity(network):
    capacity_lines = pd.Series(network.lines.s_nom_opt.values,
                               pd.MultiIndex.from_tuples([('Line', idx) for idx in network.lines.s_nom_opt.index],
                               names=['component', 'branch_i']))
    capacity_links = pd.Series(network.links.p_nom_opt.values,
                               pd.MultiIndex.from_tuples([('Link', idx) for idx in network.links.p_nom_opt.index],
                               names=['component', 'branch_i']))
    capacity = capacity_lines.append(capacity_links)
    return capacity.rename('capacity')
####################################################################################################################


# Branch Cost Model
####################################################################################################################
# €/MW*km
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

# €/MW*km
def normalised_branch_cost(network, allocation, cost_model=None, cost_factors=None):
    if cost_factors is None:
        cost_factors = branch_cost_model(allocation, cost_model=cost_model)
        
    normalised_branch_cost = cost_factors
    return normalised_branch_cost.rename('normalised branch cost')

# €
def branch_cost(network, allocation, cost_model=None, cost_factors=None, norm=None):    
    branch_cost = normalised_branch_cost(network, allocation, cost_model=cost_model, cost_factors=cost_factors) * \
                  length(network)
    
    if norm == 'MW': branch_cost = branch_cost * capacity(network)
        
    branch_cost = branch_cost.dropna()
    return branch_cost.rename('branch cost')
####################################################################################################################


# Total Transmission Cost
####################################################################################################################
def cost_normalisation(transmission_cost, total_transmission_cost):
    normalisation = total_transmission_cost / transmission_cost.sum()
    return normalisation

# €
def total_transmission_cost(network, snapshot):
    cap_lines = (network.lines['capital_cost'] * network.lines['s_nom_opt']).sum()
    
    cap_links = (network.links['capital_cost'] * network.links['p_nom_opt']).sum()
    mar_links = (network.links['marginal_cost'] * abs(network.links_t.p0.loc[snapshot])).sum().sum()

    total_cost = cap_lines + cap_links + mar_links
    return total_cost
####################################################################################################################


# Pricing Strategies
####################################################################################################################
def strategy_factor(network, allocation, pricing_strategy):
    if pricing_strategy == 'MW-Mile':
        strategy_factor = pd.Series(1.0, allocation.index)
    elif pricing_strategy == 'Capacity Pricing':
        strategy_factor = (1.0/capacity(network)).replace(np.inf, np.nan).dropna()
    else:
        raise NameError('No valid pricing strategy.')
    return strategy_factor.rename('strategy factor')
####################################################################################################################


# Main Function
####################################################################################################################
# €
def transmission_cost(network, snapshot,
                      allocation_method='Average participation',
                      pricing_strategy='MW-Mile', cost_model=None, normalisation=True, norm=None,
                      allocation=None, cost_factors=None):

    if allocation is None: allocation = netallocation.allocate_flow(network, snapshot, method=allocation_method)
        
    if cost_model is None: cost_model = pricing_strategy
        
    if cost_model == 'Capacity Pricing': norm = 'MW'
        
    transmission_cost = (abs(allocation) * \
                         strategy_factor(network, allocation, pricing_strategy)) \
                        .reorder_levels(allocation.index.names) * \
                         branch_cost(network, allocation,
                                     cost_model=cost_model, cost_factors=cost_factors, norm=norm) \
                        .reorder_levels(allocation.index.names)
    
    if normalisation == True:
        transmission_cost = transmission_cost * cost_normalisation(transmission_cost, 
                                                                   total_transmission_cost(network, snapshot))
        
    return transmission_cost.rename('transmission cost')
####################################################################################################################
