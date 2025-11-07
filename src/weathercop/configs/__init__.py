"""Configuration modules for WeatherCop."""

from . import dwd_vg_conf


def get_dwd_vg_config():
    """Return the DWD VARWG configuration module.

    This configuration defines distribution families and seasonal parameters
    for German Weather Service (DWD) meteorological variables:
    - theta (temperature)
    - R (precipitation)
    - sun (sunshine duration)
    - rh (relative humidity)

    Returns
    -------
    module
        The dwd_vg_conf module with all configuration attributes.

    Example
    -------
    >>> from weathercop.configs import get_dwd_vg_config
    >>> vg_conf = get_dwd_vg_config()
    >>> from weathercop.multisite import set_conf
    >>> set_conf(vg_conf)
    """
    return dwd_vg_conf


__all__ = ["dwd_vg_conf", "get_dwd_vg_config"]
