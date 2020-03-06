#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:13:56 2020

@author: fabian
"""

import pypsa
import pandas as pd
import netallocation as ntl
from  netallocation.cost import (nodal_co2_cost, nodal_demand_cost,
                                 nodal_production_revenue, congestion_revenue)
from netallocation.utils import get_ext_branches_i
from xarray.testing import assert_allclose, assert_equal
from pypsa.descriptors import nominal_attrs
close = lambda d1, d2: d1.round(0) == d2.round(0)


def check_duality(n, co2_constr_name=None):
    O = n.objective + ntl.cost.objective_constant(n)
    PR = nodal_production_revenue(n).sum()
    CO2C = nodal_co2_cost(n, co2_constr_name=co2_constr_name).sum()
    DC = nodal_demand_cost(n).sum()
    CR_ext, CR_fix = congestion_revenue(n, split=True)
    assert close(O, PR - CO2C - CR_ext.sum())
    assert close(O, DC - CO2C + CR_fix.sum())

def check_zero_profit_branches(n):
    cost_ext = n.branches().fillna(0).eval('(s_nom_opt + p_nom_opt) * capital_cost')\
              [get_ext_branches_i(n)]
    CR_ext, CR_fix = congestion_revenue(n, split=True)
    assert all(close(cost_ext, -CR_ext.sum('snapshot'))), ("Zero Profit "
            "condition for branches is not fulfilled.")


def test_duality_wo_investment():
    n = ntl.test.get_network_mini()
    for c, attr in nominal_attrs.items():
        n.df(c)[attr + '_extendable'] = False
    n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
    check_duality(n)
    check_zero_profit_branches(n)


def test_duality_with_investment():
    n = ntl.test.get_network_ac_dc()
    n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
    check_duality(n, co2_constr_name='co2_limit')
    check_zero_profit_branches(n)

# def test_duality_with_investment():
#     n = ntl.test.get_network_ac_dc()
#     n.generators.loc[['Manchester Wind', 'Manchester Gas'],
#                      'p_nom_extendable'] = False
#     n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
#     check_duality(n, co2_constr_name='co2_limit')
#     check_zero_profit_branches(n)


def test_duality_investment_mix_ext_nonext_lines():
    n = ntl.test.get_network_ac_dc()
    n.lines.loc['0', 's_nom_extendable'] = False
    n.lopf(pyomo=False, keep_shadowprices=True, solver_name='cbc')
    check_duality(n, co2_constr_name='co2_limit')
    check_zero_profit_branches(n)
