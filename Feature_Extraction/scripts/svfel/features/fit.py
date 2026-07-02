# -*- coding: utf-8 -*-

# linear regression
# poly?
# mean squared error
# r2 coefficient of determination (only for linear?)

import numpy as np


def least_squares(Y, x=None, order=1):
    # least squares linear fit
    # aggregrate, time_steps
    # x
    if x is None:
        x = np.arange(Y.shape[-1])
    # fit
    coef = np.polyfit(x, Y.T, deg=order).T
    return coef


def rmse(Y, Y_hat):
    # mean squared error
    # aggregrate, time_steps
    return np.sqrt(np.mean((Y - Y_hat)**2, axis=-1))
    
def r2():
    ...
    
    
    
def draw(x, coef):
    if coef.shape[-1] == 2: 
        func = _o1_eq
    elif coef.shape[-1] == 3: 
        func = _o2_eq
    elif coef.shape[-1] == 4: 
        func = _o3_eq
    elif coef.shape[-1] == 5: 
        func = _o4_eq   
    elif coef.shape[-1] == 6: 
        func = _o5_eq
    elif coef.shape[-1] == 7: 
        func = _o6_eq
    elif coef.shape[-1] == 8: 
        func = _o7_eq
    elif coef.shape[-1] == 9: 
        func = _o8_eq
    elif coef.shape[-1] == 10: 
        func = _o9_eq
    else: 
        print('NOT IMPLEMENTED')
    y_hat = []
    for c in coef:
        y_hat.append(func(x, c))
    y_hat = np.stack(y_hat)
    return y_hat.T    
    

def _o1_eq(x, c):
    return x*c[0] + c[1]

def _o2_eq(x, c):
    return (x**2)*c[0] + _o1_eq(x, c[1:])

def _o3_eq(x, c):
    return (x**3)*c[0] + _o2_eq(x, c[1:])

def _o4_eq(x, c):
    return (x**4)*c[0] + _o3_eq(x, c[1:])

def _o5_eq(x, c):
    return (x**5)*c[0] + _o4_eq(x, c[1:])

def _o6_eq(x, c):
    return (x**6)*c[0] + _o5_eq(x, c[1:])

def _o7_eq(x, c):
    return (x**7)*c[0] + _o6_eq(x, c[1:])

def _o8_eq(x, c):
    return (x**8)*c[0] + _o7_eq(x, c[1:])

def _o9_eq(x, c):
    return (x**9)*c[0] + _o8_eq(x, c[1:])    

