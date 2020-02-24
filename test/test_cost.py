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


n = ntl.test.get_network_mini()

def test_simple_wo_investment():
    # Simple network without investment
    O = n.objective
    PR = nodal_production_revenue(n).sum()
    DC = nodal_demand_cost(n).sum()
    CR = congestion_revenue(n).sum()
    assert O == PR
    assert O == DC + CR

# def test_simple_wo_investment_modified_sn_weightings():
#     # modify sn weightings
#     n.snapshot_weightings[:] = 3
#     n.lopf(solver_name='cbc')
#     O = n.objective
#     PR = nodal_production_revenue(n).sum()
#     DC = nodal_demand_cost(n).sum()
#     CR = congestion_revenue(n).sum()
#     assert O == PR
#     assert O == DC + CR
