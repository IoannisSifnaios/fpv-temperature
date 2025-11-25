"""FPV temperature model from Hayibo."""

def hayibo(temp_air, poa_global, temp_water):
    """
    Estimate floating PV module temperature using the Hayibo empirical model.

    The empirical regression model predicts module temperature as a linear
    function of ambient air temperature, water temperature, and plane-of-array
    irradiance. The model was derived from one day of experimental measurements
    from a foam float FPV systems in the USA.

    Parameters
    ----------
    temp_air : numeric
        Ambient air temperature [°C].
    poa : numeric
        Plane-of-array irradiance [W/m²].
    temp_water : numeric
        Water surface temperature underneath the PV modules [°C].

    Returns
    -------
    numeric
        Estimated module temperature [°C].

    References
    ----------
    .. [1] K. Hayibo,"Quantifying the value of foam-based flexible floating
       solar photovoltaic systems," Master's thesis, Michigan Technological
       University, 2021.
       `<https://digitalcommons.mtu.edu/etdr/1176/>`_
    """
    temp = (-13.2554 - 0.0875 * temp_water + 1.2645 * temp_air
                + 0.0128 * poa_global)
    return temp
