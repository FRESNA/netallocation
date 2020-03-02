#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:51:53 2019

@author: fabian
"""

import pypsa
import os
from pathlib import Path

modulepath = Path(__file__).parent

def get_network_ac_dc():
    return pypsa.Network(str(modulepath.joinpath('networks','ac_dc.h5')))


def get_network_pure_dc_link():
    return pypsa.Network(str(modulepath.joinpath('networks','simple_dc_model.h5')))

def get_network_large():
    return pypsa.Network(str(modulepath.joinpath('networks', 'european_model.h5')))

def get_network_mini():
    return pypsa.Network(str(modulepath.joinpath('networks', 'mini_model.h5')))
