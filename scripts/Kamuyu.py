"""FPV temperature model from Kamuyu."""

def kamuyu_1(temp_air, poa_global, wind_speed):
    """
    Estimate floating PV module temperature using the Kamuyu empirical model.

    The empirical regression model from Kamuyu [1]_ predicts module temperature
    from ambient air temperature, plane-of-array irradiance, and wind speed.
    The model was derived from experiemntal data from a pontoon float FPV
    system in South Korea.

    Parameters
    ----------
    temp_air : numeric
        Ambient air temperature [°C].
    poa_global : numeric
        Plane-of-array global irradiance [W/m²].
    wind_speed : numeric
        Wind speed at the module [m/s].

    Returns
    -------
    numeric
        Estimated module temperature [°C].

    References
    ----------
    .. [1] W. C. L. Kamuyu, J. R. Lim, C. S. Won, and H. K. Ahn,  
           "Prediction Model of Photovoltaic Module Temperature for Power
           Performance  of Floating PVs," Energies, vol. 11, no. 2, Article
           447, 2018. :doi:`10.3390/en11020447`
    """
    temp_fpv = (2.0458 + 0.9458 * temp_air + 0.0215 * poa_global
                - 1.2376 * wind_speed)
    return temp_fpv


def kamuyu_2(temp_air, poa_global, wind_speed, temp_water):
    """
    Estimate floating PV module temperature using the Kamuyu empirical model.

    The empirical regression model from Kamuyu [1]_ predicts module temperature
    from ambient air temperature, plane-of-array irradiance, wind speed, and
    water temperature. The model was derived from experiemntal data from a
    pontoon float FPV system in South Korea.

    Parameters
    ----------
    temp_air : numeric
        Ambient air temperature [°C].
    poa_global : numeric
        Plane-of-array global irradiance [W/m²].
    wind_speed : numeric
        Wind speed at the module [m/s].
    temp_water : numeric
        Temperature of the water underneath the panels [°C].

    Returns
    -------
    numeric
        Estimated module temperature [°C].

    References
    ----------
    .. [1] W. C. L. Kamuyu, J. R. Lim, C. S. Won, and H. K. Ahn,  
           "Prediction Model of Photovoltaic Module Temperature for Power
           Performance  of Floating PVs," Energies, vol. 11, no. 2, Article
           447, 2018. :doi:`10.3390/en11020447`
    """
    temp_fpv = (1.8081 + 0.9282 * temp_air + 0.021 * poa_global
                - 1.221 * wind_speed + 0.0246 * temp_water)
    return temp_fpv
