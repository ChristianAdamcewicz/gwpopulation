from importlib import import_module

__backend__ = ""
SUPPORTED_BACKENDS = ["numpy", "cupy", "jax"]
_np_module = dict(numpy="numpy", cupy="cupy", jax="jax.numpy")
_scipy_module = dict(numpy="scipy", cupy="cupyx.scipy", jax="jax.scipy")

__all__ = [
    "SUPPORTED_BACKENDS",
    "disable_cupy",
    "enable_cupy",
    "modules_to_update",
    "set_backend",
]

__doc__ = f"""
:code:`GWPopulation` provides a unified interface to a number of :code:`numpy/scipy` like APIs.

The backend can be set using :code:`gwpopulation.set_backend(backend)`, where
:code:`backend` is one of :code:`{', '.join(SUPPORTED_BACKENDS)}`.

Downstream packages can automatically track the active backend using :code:`entry_points`.
With this set up, packages can use :code:`xp` and :code:`scs` in specified modules.
Additionally, users can provide a full arbitrary scipy object to be used if anything beyond
:code:`scipy.special` is needed.
An example of how to set :code:`numpy`, :code:`scipy.special`, and the :code:`toeplitz` function
from :code:`scipy.linalg` via the :code:`setup.cfg` file is shown below.
Specification using :code:`pyproject.toml` and :code:`setup.py` follows slightly
different syntax documentation for which can be found online.

.. code-block::

    [options.entry_points]
    gwpopulation.xp =
        mypackage_foo = mypackage.foo
    gwpopulation.scs =
        mypackage_foo = mypackage.foo
        mypackage_bar = mypackage.bar
    gwpopulation.other =
        mypackage_baz_toeplitz = mypackage.baz:scipy.linalg.toeplitz

.. note::
    Each module that wants to use the :code:`GWPopulation` backend must be specified independently
    for the automatic propagation to work.

If there is a backend that you would like to use that is not currently supported, please open an issue.
"""


def modules_to_update():
    """
    Return all modules that need to be updated with the backend.

    Returns
    -------
    all_with_xp: list
        List of all modules that need to be updated with the backend's :code:`xp`
        (:code:`numpy`).
    all_with_scs: list
        List of all modules that need to be updated with the backend's :code:`scs`
        (:code:`scipy.special`).
    others: dict
        Dictionary of all modules that need to be updated with arbitrary functions
        from :code:`scipy`.
    """
    import sys

    if sys.version_info < (3, 10):
        from importlib_metadata import entry_points
    else:
        from importlib.metadata import entry_points
    all_with_xp = [module.value for module in entry_points(group="gwpopulation.xp")]
    all_with_scs = [module.value for module in entry_points(group="gwpopulation.scs")]
    all_with_sci = [module.value for module in entry_points(group="gwpopulation.sci")]
    other_entries = [
        module.value.split(":") for module in entry_points(group="gwpopulation.other")
    ]
    others = {key: value for key, value in other_entries}
    return all_with_xp, all_with_scs, all_with_sci, others


def disable_cupy():
    """
    .. deprecated:: 1.0.0

    Set the backend to :code:`numpy`.
    This function is a relic of when only :code:`numpy` and :code:`cupy`
    were supported and has been deprecated and will be removed in :code:`v1.2.0`.
    """
    from warnings import warn

    warn(
        f"Function enable_cupy is deprecated, use set_backed('cupy') instead",
        DeprecationWarning,
    )
    set_backend(backend="numpy")


def enable_cupy():
    """
    .. deprecated:: 1.0.0

    Set the backend to :code:`cupy`.
    This function is a relic of when only :code:`numpy` and :code:`cupy`
    were supported and has been deprecated and will be removed in :code:`v1.2.0`.
    """
    from warnings import warn

    warn(
        f"Function enable_cupy is deprecated, use set_backed('cupy') instead",
        DeprecationWarning,
    )
    set_backend(backend="cupy")


def _configure_jax(xp, sci):
    """
    Configuration requirements for :code:`jax`

    - use 64-bit floats.
    - update :code:`xp.trapz` to :code:`jax.scipy.integrate.trapezoid`
    """
    from jax import config
    from jax.scipy.integrate import trapezoid
    
    def _tupleset(t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)
    def _cumtrapz(y, x=None, axis=-1, initial=0):
        """
        Adapted from https://github.com/scipy/scipy/blob/v0.14.0/scipy/integrate/quadrature.py#L193
        """
        y = xp.asarray(y)
        x = xp.asarray(x)
        d = xp.diff(x)
        # reshape to correct shape
        shape = [1] * y.ndim
        shape[axis] = -1
        d = d.reshape(shape)
        nd = len(y.shape)
        slice1 = _tupleset((slice(None),)*nd, axis, slice(1, None))
        slice2 = _tupleset((slice(None),)*nd, axis, slice(None, -1))
        res = xp.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)
        shape = list(res.shape)
        shape[axis] = 1
        res = xp.concatenate([xp.ones(shape, dtype=res.dtype) * initial, res],
                            axis=axis)
        return res
        
    config.update("jax_enable_x64", True)
    xp.trapz = trapezoid
    sci.cumtrapz = _cumtrapz


def _configure_cupy(xp, sci):
    """
    Configuration requirements for :code:`cupy`
    """
    
    def _tupleset(t, i, value):
        l = list(t)
        l[i] = value
        return tuple(l)
    def _cumtrapz(y, x=None, axis=-1, initial=0):
        """
        Adapted from https://github.com/scipy/scipy/blob/v0.14.0/scipy/integrate/quadrature.py#L193
        """
        y = xp.asarray(y)
        x = xp.asarray(x)
        d = xp.diff(x)
        # reshape to correct shape
        shape = [1] * y.ndim
        shape[axis] = -1
        d = d.reshape(shape)
        nd = len(y.shape)
        slice1 = _tupleset((slice(None),)*nd, axis, slice(1, None))
        slice2 = _tupleset((slice(None),)*nd, axis, slice(None, -1))
        res = xp.cumsum(d * (y[slice1] + y[slice2]) / 2.0, axis=axis)
        shape = list(res.shape)
        shape[axis] = 1
        res = xp.concatenate([xp.ones(shape, dtype=res.dtype) * initial, res],
                            axis=axis)
        return res
        
    sci.cumtrapz = _cumtrapz


def _load_numpy_and_scipy(backend):
    try:
        xp = import_module(_np_module[backend])
        scs = import_module(_scipy_module[backend]).special
        if backend == "cupy":
            sci = import_module(_scipy_module[backend])
        else:
            sci = import_module(_scipy_module[backend]).integrate
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"{backend} not installed for gwpopulation")
    except ImportError:
        raise ImportError(f"{backend} installed but not importable for gwpopulation")

    if backend == "jax":
        _configure_jax(xp, sci)
    if backend == "cupy":
        _configure_cupy(xp, sci)

    return xp, scs, sci


def _load_arbitrary(func, backend):
    if func.startswith("scipy"):
        func = func.replace("scipy", _scipy_module[backend])
    elif func.startswith("numpy"):
        func = func.replace("numpy", _np_module[backend])
    module, func = func.rsplit(".", 1)
    return getattr(import_module(module), func)


def set_backend(backend="numpy"):
    """
    Set the backend for :code:`GWPopulation` and plugins.

    This will automatically update all modules that have been registered to use
    the :code:`GWPopulation` automatic backend tracking.

    .. warning::

        This will not update existing instances of classes.

    Parameters
    ----------
    backend: str
        The backend to use, one of the :code:`SUPPORTED_BACKENDS`.

    Raises
    ------
    ValueError
        If the backend is not in :code:`SUPPORTED_BACKENDS`.
    """
    global __backend__
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Backend {backend} not supported, should be in {', '.join(SUPPORTED_BACKENDS)}"
        )
    elif backend == __backend__:
        return

    xp, scs, sci = _load_numpy_and_scipy(backend)

    __backend__ = backend
    all_with_xp, all_with_scs, all_with_sci, others = modules_to_update()
    for module in all_with_xp:
        setattr(import_module(module), "xp", xp)
    for module in all_with_scs:
        setattr(import_module(module), "scs", scs)
    for module in all_with_sci:
        setattr(import_module(module), "sci", sci)
    for module, func in others.items():
        setattr(
            import_module(module),
            func.split(".")[-1],
            _load_arbitrary(func, backend),
        )
