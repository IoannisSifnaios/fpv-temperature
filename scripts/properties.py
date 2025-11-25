"""Fluid properties and functions related to weather parameters."""

import numpy as np


def C_to_K(temp):
    """
    Convert temperaure from degrees Celsius to Kelvin.

    Parameters
    ----------
    temp : numeric
        Temperature in degrees Celsius [°C].

    Returns
    -------
    Temperature in Kelvin [K]
    """
    return temp + 273.15


def K_to_C(temp):
    """
    Convert temperaure from Kelvin to degrees Celsius.

    Parameters
    ----------
    temp : numeric
        Temperature in Kelvin [K].

    Returns
    -------
    Temperature in degrees Celsius [°C]
    """
    return temp - 273.15


def wind_speed_power_law(wind_speed_reference, height_reference,
                         height_desired, exponent=1/5):
    """
    Estimate wind speed at a difference height using the power law.

    Parameters
    ----------
    wind_speed_reference : numeric
        Wind speed at the reference height.
    height_reference : numeric
        Height of the reference wind speed measurements.
    height_desired : numeric
        Height for which to estimate wind speed. Must have the same units
        as ``height_reference``.
    exponent : optional
        Exponent used in the power law to estimate wind speed. The default
        is 1/5. Some commonly used values are:

        - 1/7: offshore
        - 1/5: neutral stability over open land
        - 0.3–0.4: urban terrain

    Returns
    -------
    Estimate of wind speed. The unit is the same as the
    ``wind_speed_reference``.
    """
    return wind_speed_reference * (height_desired / height_reference)**exponent


def get_thermal_conductivity(temp, medium):
    """
    Estimate thermal conductivity.

    Equations from [1]_ for air at atmoshperic pressure and [2]_ for water.

    Parameters
    ----------
    temp : numeric
        Temperatures for which to estimate the property [°C].
    medium : str
        The medium of interest.

    Returns
    -------
    thermal_conductivity
        Thermal conductivity [W/m/K].

    References
    ----------
    .. [1] International Electrotechnical Commission. (2024).
       IEC 60287-2-3:2024 – Electric cables – Calculation of the current rating
       – Part 2-3: Thermal resistance – Cables installed in ventilated tunnels
    .. [2] Dixon, J. C. (2007). The shock-absorber handbook (2nd ed.).
       John Wiley & Sons. Appendix C: Properties of Water.
       :doi:`10.1002/9780470516430.app3`
    """
    if medium == 'air':
        return 2.42e-2 + 7.2e-5 * temp
    elif medium == 'water':
        return 0.5706 + 1.756e-3 * temp - 6.46e-6 * temp**2
    else:
        raise ValueError(f"{medium} is not a supported medium.")


def get_prandtl(temp, medium):
    """
    Estimate Prandtl number.

    Equations from [1]_ and [2]_.

    Parameters
    ----------
    temp : numeric
        Temperatures for which to estimate the property [°C].
    medium : str
        The medium of interest.

    Returns
    -------
    prandtl_number
        Prandtl number [-]

    References
    ----------
    .. [1] `Wikipedia page on Prandtl number
       <https://en.wikipedia.org/wiki/Prandtl_number>`_
    .. [2] `Prandtl number, tec-science.com
       <https://www.tec-science.com/mechanics/gases-and-liquids/prandtl-number/>`_
    """
    if medium == 'air':
        if (np.max(temp) >= 500) or (np.min(temp) <= -100):
            raise ValueError("Input temperature is outside acceptable limits.")
        return 10**9 / (
            1.1*temp**3 - 1200*temp**2 + 322000*temp + 1.393*10**9)
    elif medium == 'water':
        if (np.max(temp) >= 90) or (np.min(temp) <= 0):
            raise ValueError("Input temperature is outside acceptable limits.")
        return 50000 / (temp**2 + 155*temp + 3700)
    else:
        raise ValueError(f"{medium} is not a supported medium.")


def get_dynamic_viscosity(temp, medium):
    """
    Estimate dynamic viscosity.

    Equations from [1]_.

    Parameters
    ----------
    temp : numeric
        Temperatures for which to estimate the property [°C].
    medium : str
        The medium of interest.

    Returns
    -------
    dynamic_viscosity
        Dynamic viscosity [Pa·s]

    References
    ----------
    .. [1] `Wikipedia page on viscosity
       <https://en.wikipedia.org/wiki/Viscosity>`_
    """
    # These expressions require the temperature to be in [K]
    temp_K = C_to_K(temp)

    if medium == 'air':
        return 2.791e-7 * temp_K**0.7355
    elif medium == 'water':
        return 2.939e-5 * np.exp(507.88 / (temp_K - 149.3))
    else:
        raise ValueError(f"{medium} is not a supported medium.")


def get_density(temp, medium):
    """
    Estimate density.

    Equations from [1]_ for air and [2]_ for water.

    Parameters
    ----------
    temp : numeric
        Temperatures for which to estimate the property [°C].
    medium : str
        The medium of interest.

    Returns
    -------
    density
        Density [kg/m³]

    References
    ----------
    .. [1] `Wikipedia page on density of air
       <https://en.wikipedia.org/wiki/Density_of_air>`_
    .. [2] Dixon, J. C. (2007). The shock-absorber handbook (2nd ed.).
       John Wiley & Sons. Appendix C: Properties of Water.
       :doi:`10.1002/9780470516430.app3`
    """
    # The expression for air requires the temperature to be in [K]
    temp_K = C_to_K(temp)

    if medium == 'air':
        return 101325 / (287.05 * temp_K)
    elif medium == 'water':
        return 1001.3 - 0.155 * temp - 2.658e-3 * temp**2
    else:
        raise ValueError(f"{medium} is not a supported medium.")


def get_cp(temp, medium):
    """
    Estimate specific heat.

    Equations from [1]_.

    Parameters
    ----------
    temp : numeric
        Temperatures for which to estimate the property [°C].
    medium : str
        The medium of interest.

    Returns
    -------
    density
        Specific heat [J/(kg·K)]

    References
    ----------
    .. [1] McBride, B. J., Zehe, M. J., and Gordon S. (2002). NASA Glenn
       Coefficients for Calculating Thermodynamic Properties of Individual
       Species <https://ntrs.nasa.gov/citations/20020085330>`_
    """
    # The expression for air requires the temperature to be in [K]
    temp_K = C_to_K(temp)

    if medium == 'air':
        Cp = (1.009950160e4 * temp_K**(-2)
              - 1.968275610e2 * temp_K**(-1)
              + 5.009155110
              - 5.761013730e-3 * temp_K
              + 1.066859930e-5 * temp_K**2
              - 7.940297970e-9 * temp_K**3
              + 2.185231910e-12 * temp_K**4) * 8.314510  # J/(mol K)

        M = 0.0289651159  # kg/mol (molar mass of dry air)

    elif medium == 'water':
        Cp = (-3.947960830e4 * temp_K**(-2)
              + 5.755731020e2 * temp_K**(-1)
              + 9.317826530e-1
              + 7.222712860e-3 * temp_K
              - 7.342557370e-6 * temp_K**2
              + 4.955043490e-9 * temp_K**3
              - 1.336933246e-12 * temp_K**4) * 8.314510  # J/(mol K)

        M = 0.01801528  # kg/mol (molar mass of water)

    else:
        raise ValueError(f"{medium} is not a supported medium.")

    return Cp / M
