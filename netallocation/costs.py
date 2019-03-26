import numpy as np
import pandas as pd

from . import allocate_flow


# Branch Cost Model
####################################################################################################################
# Extracts the length values of lines and links specified in a given PyPSA network
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

# Here one may determine a model for costs per unit length of lines and links specified in a given PyPSA network
def branch_cost_model(allocation, cost_model=None):
    # Usually the cost for a given branch are constant, however
    # one may penalise some countries and their transactions
    # Note: np.random allocates purely random costs to each single transaction
    if cost_model is None:
        raise NameError('No valid cost model for the given pricing strategy.')
    # €/MW*km
    elif cost_model == 'MW-Mile':
        cost_factors = np.random.random(len(allocation.index))
    # €/km
    elif cost_model == 'Capacity Pricing':
        cost_factors = np.random.random(len(allocation.index))
    # €/MW*km
    elif cost_model == 'Random Numbers':
        cost_factors = np.random.random(len(allocation.index))
    # €/MW*km
    elif cost_model == 'Unit Test':
        cost_factors = 1.0
    else:
        raise NameError('No valid cost model for the given pricing strategy.')
    return cost_factors

# If cost factors are already available, one may skip the branch_cost_model routine by
# inserting the cost factors directly
# €/MW*km
def normalised_branch_cost(allocation, cost_model=None, cost_factors=None):
    if cost_factors is None: cost_factors = branch_cost_model(allocation, cost_model=cost_model)
    normalised_branch_cost = pd.Series(cost_factors, allocation.index,
                                       name='normalised branch cost')
    return normalised_branch_cost

# Calculates the branch cost of each allocated transaction by multiplying the costs per unit length times the length
# €/MW
def branch_cost(network, allocation, cost_model=None, cost_factors=None):    
    branch_cost = normalised_branch_cost(allocation, cost_model=cost_model, cost_factors=cost_factors) * \
                  length(network)
    branch_cost = branch_cost.dropna().reorder_levels(['snapshot', 'source', 'sink', 'component', 'branch_i'])
    # Drop lines/links that show no flow, i.e. do not take part in the flow allocation procedure at all
    return branch_cost.rename('branch cost')
####################################################################################################################


# Pricing Strategies
####################################################################################################################
def capacity(network):
    capacity_lines = pd.Series(network.lines.s_nom_opt.values,
                               pd.MultiIndex.from_tuples([('Line', idx) for idx in network.lines.s_nom_opt.index],
                               names=['component', 'branch_i']))
    capacity_links = pd.Series(network.links.p_nom_opt.values,
                               pd.MultiIndex.from_tuples([('Link', idx) for idx in network.links.p_nom_opt.index],
                               names=['component', 'branch_i']))
    capacity = capacity_lines.append(capacity_links)
    return capacity.rename('capacity')

# Implements MW-Mile
# €
def mw_mile(allocation, branch_cost):
    transmission_cost = abs(allocation)*branch_cost
    return transmission_cost.rename('transmission cost')

# Implements Capacity Pricing
# €
def capacity_pricing(network, allocation, branch_cost):
    transmission_cost = (abs(allocation)/capacity(network)) \
                        .dropna().reorder_levels(['snapshot', 'source', 'sink', 'component', 'branch_i']) * \
                        branch_cost
    return transmission_cost.rename('transmission cost')
####################################################################################################################


# Total System Costs
####################################################################################################################
# Extracts optimised capital and marginal cost of lines and links for a given PyPSA network
# €
def total_cost(network, snapshot):
    
    return total_cost


# Main Function
####################################################################################################################
# Calculates transmission cost using different pricing strategies
# Transmission cost does not include infrastructure, i.e. maintainance, reliability, etc.
# €
def transmission_cost(network, snapshot,
                      allocation_method='Average participation',
                      pricing_strategy='MW-Mile',
                      cost_model=None,
                      allocation=None,
                      cost_factors=None):

    # If no allocation is given, the specified allocation routine is executed
    if allocation is None: allocation = netallocation.allocate_flow(network, snapshot, method=allocation_method)
        
    # If no cost model is specified, it will be determined by the actual pricing strategy
    if cost_model is None: cost_model = pricing_strategy
    
    if pricing_strategy == 'MW-Mile':
        transmission_cost = mw_mile(allocation,
                                    branch_cost(network, allocation,
                                                cost_model=cost_model, cost_factors=cost_factors))
        
    elif pricing_strategy == 'Capacity Pricing':
        transmission_cost = capacity_pricing(network, allocation,
                                             branch_cost(network, allocation,
                                                         cost_model=cost_model, cost_factors=cost_factors))
    else:
        raise NameError('No valid pricing strategy.')
        
    return transmission_cost
####################################################################################################################
