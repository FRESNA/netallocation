#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:11:43 2019

@author: fabian
"""



def optimal_flow_shares(n, snapshot, method='min', downstream=True,
                        per_bus=False, **kwargs):
    """



    """
    from scipy.optimize import minimize
    H = PTDF(n)
    p = n.buses_t.p.loc[snapshot]
    p_plus = p.clip(lower=0)
    p_minus = p.clip(upper=0)
    pp = p.to_frame().dot(p.to_frame().T).div(p).fillna(0)
    if downstream:
        indiag = diag(p_plus)
        offdiag = (p_minus.to_frame().dot(p_plus.to_frame().T)
                   .div(p_plus.sum()))
        pp = pp.clip(upper=0).add(diag(pp)).mul(np.sign(p.clip(lower=0)))
        bounds = pd.concat([pp.stack(), pp.stack().clip(lower=0)], axis=1,
                           keys=['lb', 'ub'])

#                   .pipe(lambda df: df - np.diagflat(np.diag(df)))
    else:
        indiag = diag(p_minus)
        offdiag = (p_plus.to_frame().dot(p_minus.to_frame().T)
                   .div(p_minus.sum()))
        pp = pp.clip(lower=0).add(diag(pp)).mul(-np.sign(p.clip(upper=0)))
        bounds = pd.concat([pp.stack().clip(upper=0), pp.stack()], axis=1,
                           keys=['lb', 'ub'])
    x0 = (indiag + offdiag).stack()
    N = len(n.buses)
    if method == 'min':
        sign = 1
    elif method == 'max':
        sign = -1

    def minimization(df):
        return sign * (H.dot(df.reshape(N, N)).stack()**2).sum()

    constr = [
            #   nodal balancing
            {'type': 'eq', 'fun': lambda df: df.reshape(N, N).sum(0)},
            #    total injection of colors
            {'type': 'eq', 'fun': lambda df: df.reshape(N, N).sum(1)-p.values}
            ]

    #   sources-sinks-fixation
    res = minimize(minimization, x0, constraints=constr,
                   bounds=bounds, options={'maxiter': 1000}, tol=1e-5,
                   method='SLSQP')
    print(res)
    sol = pd.DataFrame(res.x.reshape(N, N), columns=n.buses.index,
                       index=n.buses.index).round(10)
    if per_bus:
        return (sol[indiag.sum()==0].T
                .rename_axis('sink', axis=int(downstream))
                .rename_axis('source', axis=int(not downstream))
                .stack()[lambda ds:ds != 0])
    else:
        return H.dot(sol).round(8)

