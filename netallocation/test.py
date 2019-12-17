#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:51:53 2019

@author: fabian
"""

import pypsa
import os

def get_network_ac_dc():
    return pypsa.Network(os.path.join(__file__, '..', '..', 'test', 'networks','test.nc'))


def get_network_pure_dc_link():
    return pypsa.Network(os.path.join(__file__, '..', '..', 'test', 'networks','simple_dc_model.nc'))

def get_network_large():
    return pypsa.Network(os.path.join(__file__, '..', '..', 'test', 'networks', 'european_model.nc'))

