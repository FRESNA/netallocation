import numpy as np
from numpy import sign
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
                    snapshot_weightings, split_one_ports,
                    get_as_dense_by_bus_carrier)
from .linalg import norm
from .convert import vip_to_p2p, virtual_patterns
from .breakdown import expand_by_source_type
from .grid import (Incidence, impedance, energy_production, energy_demand,
                   power_production)

logger = logging.getLogger(__name__)


def allocate_one_port_operational_cost(ds, n, snapshots=None, dim='source'):
    """
    Allocate production costs on the basis of a peer-to-peer allocation.

    Parameters
    ----------
    ds : xarray.Dataset
        Calculated power allocation dataset, i.e. from ntl.allocate_flow.
    n : pypsa.Network
    dim : str, default 'source'
        Dimension to whicht the production costs are associated to, (must
        contain bus names).

    Returns
    -------
    xr.DataArray
        Allocated generation cost.

    """
    snapshots = check_snapshots(ds.snapshot, n)
    check_carriers(n)
    ds = expand_by_source_type(ds, n, dim=dim)
    comps = ['Generator', 'StorageUnit', 'Store']
    mc = get_as_dense_by_bus_carrier(n, 'marginal_cost', comps, snapshots)\
         .rename(bus=dim, carrier='source_carrier')
    attr = {'payer': dim, 'allocation': 'one_port_operational_cost'}
    return (mc * ds).assign_attrs(attr)


def allocate_co2_cost(ds, n, dim='source', co2_constr_name=None,
                      co2_attr='co2_emissions'):
    """
    Allocate CO2 emission costs on the basis of a peer-to-peer allocation.

    Parameters
    ----------
    ds : xarray.Dataset
        Calculated power allocation dataset, i.e. from ntl.allocate_flow.
    n : pypsa.Network
    dim : str, default 'source'
        Dimension to whicht the production costs are associated to, (must
        contain bus names).

    Returns
    -------
    xr.DataArray
        Allocated generation cost.

    """
    snapshots = check_snapshots(ds.snapshot, n)
    check_carriers(n)
    ds = expand_by_source_type(ds, n, dim=dim)
    ep = nodal_co2_price(n, snapshots, co2_attr, co2_constr_name)\
          .rename(bus=dim, carrier='source_carrier')
    ep = ep * snapshot_weightings(n, snapshots)
    attr = {'payer': dim, 'allocation': 'co2_cost'}
    return (ep * ds).assign_attrs(attr)


def allocate_one_port_investment_cost(ds, n, dim='source', proportional=False):
    """
    Allocate investment costs on the basis of a peer-to-peer allocation.

    Parameters
    ----------
    ds : xarray.Dataset
        Calculated power allocation dataset, i.e. from ntl.allocate_flow.
    n : pypsa.Network
    dim : str, default 'source'
        Dimension to whicht the investment costs are associated to, (must
        contain bus names).

    Returns
    -------
    xr.DataArray
        Allocated generation cost.

    """
    check_carriers(n)
    ds = expand_by_source_type(ds, n, dim=dim)

    comps = ['Generator']
    attr = nominal_attrs
    nom_opt = concat([reindex_by_bus_carrier(n.df(c)[attr[c] + "_opt"], c, n)
                      for c in comps], dim='carrier')
    cap_cost = concat((reindex_by_bus_carrier(n.df(c).capital_cost, c, n)
               for c in comps), dim='carrier')
    investment_cost = (nom_opt * cap_cost).rename(bus=dim, carrier='source_carrier')

    prod = power_production(n, per_carrier=True)\
            .rename(bus=dim, carrier='source_carrier')

    if not proportional:
        c = 'Generator'
        mu_upper = n.pnl(c).mu_upper
        scaling = (reindex_by_bus_carrier(mu_upper, c, n)
                   .rename(bus=dim, carrier='source_carrier'))
        prod *= scaling.reindex_like(prod, fill_value=0)
        ds *= scaling.reindex_like(ds, fill_value=0)

    normed = (ds / prod.sum('snapshot')).fillna(0)


    attr = {'payer': dim, 'allocation': 'one_port_investment_cost'}
    return (investment_cost.reindex_like(normed, fill_value=0) * normed)\
            .assign_attrs(attr)


def allocate_branch_operational_cost(ds, n):
    """
    Allocate the branch cost on the basis of an allocation method.

    Parameters
    ----------
    ds : xarray.Dataset
        Calculated power allocation dataset, i.e. from ntl.allocate_flow.
    n : pypsa.Network

    Returns
    -------
    xr.DataArray
        Allocated branch cost.

    """
    snapshots = check_snapshots(ds.snapshot, n)
    check_carriers(n)
    branchcost_pu = pd.concat([get_as_dense(n, 'Link', 'marginal_cost', snapshots)],
                              keys=['Link'], axis=1,
                              names=['component', 'branch_i'])
    branchcost_pu = DataArray(branchcost_pu, dims=['snapshot','branch'])
    attr = {'allocation': 'branch_operational_cost'}
    return (branchcost_pu.reindex_like(ds, fill_value=0) * ds).assign_attrs(attr)



def allocate_branch_investment_cost(ds, n):
    """
    Allocate the branch cost on the basis of an allocation method.

    Parameters
    ----------
    ds : xarray.Dataset
        Calculated power allocation dataset, i.e. from ntl.allocate_flow.
    n : pypsa.Network

    Returns
    -------
    xr.DataArray
        Allocated branch cost.

    """
    check_carriers(n)
    names=['component', 'branch_i']
    nom_attr = pd.Series(nominal_attrs)[np.unique(ds.component)] + '_opt'

    flow = network_flow(n, branch_components=nom_attr.index)

    investment_cost = pd.concat({c: n.df(c).eval(f'capital_cost * {attr}')
                          for c, attr in nom_attr.items()}, names=names)
    investment_cost = DataArray(investment_cost, dims='branch')

    scaling = pd.concat({c: n.pnl(c).mu_upper - n.pnl(c).mu_lower
                       for c in nom_attr.index}, axis=1, names=names)
    scaling = DataArray(scaling, dims=['snapshot', 'branch'])

    ds = ds * scaling.reindex_like(ds)
    flow = flow * scaling

    normed = (ds / flow.sum('snapshot')).fillna(0)
    attr = {'allocation': 'branch_investment_cost'}
    return (investment_cost * normed).assign_attrs(attr)



def allocate_carrier_attribute(ds, n, attr):
    """
    Allocate an carrier attribute on the basis of a peer-to-peer allocation.

    Parameters
    ----------
    ds : xarray.Dataset
        Calculated power allocation dataset, i.e. from ntl.allocate_flow.
    n : pypsa.Network
    attr : str/pd.Series/pd.DataFrame

    Returns
    -------
    xr.DataArray

    """
    check_carriers(n)
    ds = expand_by_source_type(ds, n)
    return DataArray(n.carriers[attr], dims='source_carrier') * ds


# =============================================================================
# Power Market Quantities
# =============================================================================


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
    return (energy_demand(n, snapshots) * locational_market_price(n, snapshots))\
            .rename('nodal_demand_cost')


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
            return reindex_by_bus_carrier(pd.Series(0, n.generators.index), c, n)
        else:
            co2_constr_name = co2_constr_name[0]
    elif co2_constr_name not in n.global_constraints.index:
        logger.warning(f'Constraint {co2_constr_name} not in n.global_constraints'
                    ', setting CO₂ constraint cost to zero.')
        return reindex_by_bus_carrier(pd.Series(0, n.generators.index), c, n)
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
        return split_one_ports(cost, n).sum('carrier')
    return cost.sum('carrier').rename('nodal_co2_cost')


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
    return (energy_production(n, snapshots, per_carrier=per_carrier) * \
            locational_market_price(n, snapshots))\
            .rename('nodal_production_revenue')



def objective_constant(n):
    nom_attr = nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = get_extendable_i(n, c)
        constant += n.df(c)[attr][ext_i] @ n.df(c).capital_cost[ext_i]
    return constant


def allocate_cost(n, snapshots=None, method='ap', q=0, **kwargs):
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
    kwargs : dict
        Keyword arguments for ``flow_allocation()``. For operational and
        capital costs for branches, the allocation default to q=0, meaning
        that all transmission costs are covered by the sinks. For a 50%-50%
        split between sinks and sources, set q to 0.5. for transmission cost
        totally allocated to sources, set q to 1.

    Returns
    -------
    xarray.DataArray
        Peer-to-peer cost allocation.

    """
    if isinstance(method, str):
        ds = flow_allocation(n, snapshots, method, **kwargs)
    else:
        ds = method
    ds = vip_to_p2p(virtual_patterns(ds, n, q=q), n)
    ds = expand_by_source_type(ds, n)

    p2p = ds.peer_to_peer
    vfp = ds.virtual_flow_pattern
    op_one_port = allocate_one_port_operational_cost(p2p, n)
    co2_one_port = allocate_co2_cost(p2p, n)
    inv_one_port = allocate_one_port_investment_cost(p2p, n)
    op_branch = allocate_branch_operational_cost(vfp, n)
    inv_branch = allocate_branch_investment_cost(vfp, n)

    d = dict(sink='payer', bus='payer', branch='receiver_transmission_cost',
             source='receiver_nodal_cost', source_carrier='receiver_carrier')
    def rename(da): return da.rename({k: v for k, v in d.items() if k in da.dims})
    res = Dataset({ds.attrs['allocation']: rename(ds)
                   for ds in [op_one_port, co2_one_port, inv_one_port,
                              op_branch, inv_branch]
                   if ds.sum() != 0})
    return res


