#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 18:10:01 2020

@author: fabian
"""

from functools import wraps


def check_snapshots(func):
    """
    Decorator to set snapshots to n.snapshots if None.

    Parameters
    ----------
    func : function
        Function with network as an (positional) argument (name 'n').

    """
    argnames = func.__code__.co_varnames
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        # Convert args to keyword arguments
        for i, arg in enumerate(args):
            kwargs[argnames[i]] = arg
        if kwargs['snapshots'] is None:
            kwargs['snapshots'] = kwargs['n'].snapshots.rename('snapshot')
        return func(*(), **kwargs)
    return func_wrapper

def check_branch_components(func):
    """
    Decorator to set branch_component to n.branch_components if None.

    Parameters
    ----------
    func : function
        Function with network as an (positional) argument (name 'n').

    """
    argnames = func.__code__.co_varnames
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        # Convert args to keyword arguments
        for i, arg in enumerate(args):
            kwargs[argnames[i]] = arg
        if kwargs['branch_components'] is None:
            kwargs['branch_components'] = kwargs['n'].branch_components
        return func(*(), **kwargs)
    return func_wrapper

def check_passive_branch_components(func):
    """
    Decorator to set branch_component to n.passive_branch_components if None.

    Parameters
    ----------
    func : function
        Function with network as an (positional) argument (name 'n').

    """
    argnames = func.__code__.co_varnames
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        # Convert args to keyword arguments
        for i, arg in enumerate(args):
            kwargs[argnames[i]] = arg
        components = kwargs.get('branch_components', None)
        if components is None:
                   kwargs['branch_components'] = kwargs['n'].passive_branch_components
        return func(**kwargs)
    return func_wrapper


def check_store_carrier(func):
    """
    Decorator to define carrier of stores in a network if undefined.

    Parameters
    ----------
    func : function
        Function with network as an (positional) argument (name 'n').

    """
    argnames = func.__code__.co_varnames
    @wraps(func)
    def func_wrapper(*args, **kwargs):
        # Convert args to keyword arguments
        for i, arg in enumerate(args):
            kwargs[argnames[i]] = arg
        n = kwargs['n']
        if 'carrier' not in n.stores:
            n.stores['carrier'] = n.stores.bus.map(n.buses.carrier)
    return func_wrapper



