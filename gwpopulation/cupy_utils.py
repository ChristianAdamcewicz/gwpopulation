"""
Helper functions for missing functionality in cupy.
"""

try:
    import cupy as xp
    from cupyx.scipy.special import erf, gammaln, erfinv

    CUPY_LOADED = True
except ImportError:
    import numpy as xp
    from scipy.special import erf, gammaln, erfinv

    CUPY_LOADED = False


def betaln(alpha, beta):
    r"""
    Logarithm of the Beta function

    .. math::
        \ln B(\alpha, \beta) = \frac{\ln\gamma(\alpha)\ln\gamma(\beta)}{\ln\gamma(\alpha + \beta)}

    Parameters
    ----------
    alpha: float
        The Beta alpha parameter (:math:`\alpha`)
    beta: float
        The Beta beta parameter (:math:`\beta`)

    Returns
    -------
    ln_beta: float, array-like
        The ln Beta function

    """
    ln_beta = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
    return ln_beta


def to_numpy(array):
    """Cast any array to numpy"""
    if not CUPY_LOADED:
        return array
    else:
        return xp.asnumpy(array)


def trapz(y, x=None, dx=1.0, axis=-1):
    """
    Lifted from `numpy <https://github.com/numpy/numpy/blob/v1.15.1/numpy/lib/function_base.py#L3804-L3891>`_.

    Integrate along the given axis using the composite trapezoidal rule.
    Integrate `y` (`x`) along given axis.

    Parameters
    ==========
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    =======
    trapz : float
        Definite integral as approximated by trapezoidal rule.


    References
    ==========
    .. [1] Wikipedia page: http://en.wikipedia.org/wiki/Trapezoidal_rule

    Examples
    ========
    >>> trapz([1,2,3])
    4.0
    >>> trapz([1,2,3], x=[4,6,8])
    8.0
    >>> trapz([1,2,3], dx=2)
    8.0
    >>> a = xp.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> trapz(a, axis=0)
    array([ 1.5,  2.5,  3.5])
    >>> trapz(a, axis=1)
    array([ 2.,  8.])
    """
    y = xp.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = xp.asanyarray(x)
        if x.ndim == 1:
            d = xp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = xp.diff(x, axis=axis)
    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    product = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
    try:
        ret = product.sum(axis)
    except ValueError:
        ret = xp.add.reduce(product, axis)
    return ret


def tupleset(t, i, value):
    l = list(t)
    l[i] = value
    return tuple(l)


def cumtrapz(y, x=None, dx=1.0, axis=-1, initial=0):
    y = xp.asarray(y)
    if x is None:
        d = dx
    else:
        x = xp.asarray(x)
        if x.ndim == 1:
            d = xp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = -1
            d = d.reshape(shape)
        elif len(x.shape) != len(y.shape):
            raise ValueError("If given, shape of x must be 1-d or the "
                             "same as y.")
        else:
            d = xp.diff(x, axis=axis)

        if d.shape[axis] != y.shape[axis] - 1:
            raise ValueError("If given, length of x along axis must be the "
                             "same as y.")
        
    
    nd = len(y.shape)
    slice1 = tupleset((slice(None),)*nd, axis, slice(1, None))
    slice2 = tupleset((slice(None),)*nd, axis, slice(None, -1))
    res = xp.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)

    if initial is not None:
        if not xp.isscalar(initial):
            raise ValueError("`initial` parameter should be a scalar.")

        shape = list(res.shape)
        shape[axis] = 1
        res = xp.concatenate([xp.ones(shape, dtype=res.dtype) * initial, res],
                             axis=axis)
    
    return res


def FPMIN(arr, minval = 1.0e-30):
    arr[(arr < minval)] = minval

def betacf(a, b, x):
    MAXIT = 100
    EPS = 3.0e-7

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x/qap
    FPMIN(d)
    d = 1.0/d
    h = d

    for m in range(1, MAXIT+1):
        m2 = 2*m
        aa = m * (b-m) * x/((qam+m2) * (a+m2))
        d = 1.0 + aa*d
        FPMIN(d)
        c = 1.0 + aa/c
        FPMIN(c)
        d = 1.0/d
        h *= d*c
        aa = -(a+m) * (qab+m) * x/((a+m2) * (qap+m2))
        d = 1.0 + aa*d
        FPMIN(d)
        c = 1.0 + aa/c
        FPMIN(c)
        d = 1.0/d
        dell = d*c
        h *= dell
        if abs(xp.all(h) - 1) < EPS:
            break

    return h

def betainc(a,b,x):
    result = xp.zeros(xp.shape(x))

    cut = ((x > 0.0) & (x < 1.0))
    cut1 = ((x < 0.0) | (x > 1.0))
    cut2 = (x < (a + 1.0)/(a + b + 2.0))

    result[cut] = xp.exp(gammaln(a+b)-gammaln(a)-gammaln(b) + a*xp.log(x[cut]) + b*xp.log(1.0 - x[cut]))
    result[cut1] = xp.nan

    result[cut2] = result[cut2] * betacf(a,b,x[cut2])/a
    result[~cut2] = 1.0 - result[~cut2] * betacf(b,a,1.0 - x[~cut2])/b

    return result
