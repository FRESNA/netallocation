#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:16:50 2019

@author: fabian
"""

import pandas as pd
import pypsa
from .flow import flow_allocation as allocate_flow
from .grid import Incidence, network_flow, network_injection
from .linalg import diag, inv, pinv

pd.Series.__rshift__ = lambda d1, d2: pd.testing.assert_series_equal(
        d1.round(6), d2.round(6), check_exact=False, check_names=False)

pd.DataFrame.__rshift__ = lambda d1, d2: pd.testing.assert_frames_equal(
        d1.round(6), d2.round(6), check_exact=False, check_names=False)
