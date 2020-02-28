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
from xarray.testing import assert_allclose, assert_equal
close = lambda d1, d2: d1.round(0) == d2.round(0)



def check_duality(n, co2_constr_name=None):
    n.lopf(pyomo=False, keep_shadowprices=True, solver_name='gurobi')
    O = n.objective + ntl.cost.objective_constant(n)
    PR = nodal_production_revenue(n).sum()
    CO2C = nodal_co2_cost(n, co2_constr_name=co2_constr_name).sum()
    DC = nodal_demand_cost(n).sum()
    CR_ext, CR_fix = congestion_revenue(n, split=True)
    assert close(O, PR - CO2C - CR_ext.sum())
    assert close(O, DC - CO2C + CR_fix.sum())



def test_simple_wo_investment():
    # Simple network without investment
    n = ntl.test.get_network_mini()
    check_duality(n)
