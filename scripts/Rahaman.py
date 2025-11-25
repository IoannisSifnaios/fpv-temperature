"""FPV temperature models from Rahaman."""

import numpy as np
from properties import (get_thermal_conductivity, get_prandtl,
                        get_dynamic_viscosity, get_density, C_to_K)


def rahaman_empirical(temp_air, poa_global, ghi, relative_humidity, wind_speed,
                      wind_direction, temp_water):
    """
    Estimate floating PV module temperature using the Rahaman empirical model.

    The empirical regression model from Rahaman [1]_ predicts module
    temperature from ambient air temperature, plane-of-array irradiance,
    global horizontal irradiance, relative humidity, wind speed, wind
    direction, and water temperature. The model was derived from experimental
    data from a pure pontoon float FPV system in Brazil. The number of days
    used was not disclosed.

    Parameters
    ----------
    temp_air : numeric
        Ambient air temperature [°C].
    poa_global : numeric
        Plane-of-array global irradiance [W/m²].
    ghi : numeric
        Global horizontal irradiance  [W/m²].
    relative_humidity : numeric
        Relative humidity [%].
    wind_speed : numeric
        Wind speed at the module [m/s].
    wind_direction : numeric
        Wind direct [degrees].
    temp_water : numeric
        Temperature of the water beneath the modules.

    Returns
    -------
    numeric
        Estimated module temperature [°C].

    References
    ----------
    .. [1] M. A. Rahaman, T. L. Chambers, A. Fekih, G. Wiecheteck, G. Carranza,
           and G. R. C. Possetti, "Floating photovoltaic module temperature
           estimation: Modeling and comparison," Renewable Energy, vol. 208,
           pp. 162–180, Mar. 2023, :doi:`10.1016/j.renene.2023.03.076`.
    """
    temp_fpv = (2.052 - 0.053 * relative_humidity + 0.965527 * temp_air
                + 0.00683 * poa_global + 0.1364 * temp_water
                - 0.495 * wind_speed + 0.0028 * wind_direction + 0.0187 * ghi)
    return temp_fpv


def _convection_coefficient(
        temp_air, temp_module, wind_speed, module_height, module_width,
        module_thickness, module_tilt, module_side):

    # Convective heat transfer coefficients are eval. at the film temp.
    temp_film = (temp_air + temp_module) / 2

    # Air properties
    lamda_air = get_thermal_conductivity(temp_film, medium='air')
    Pr_air = get_prandtl(temp_film, medium='air')
    mu_air = get_dynamic_viscosity(temp_film, medium='air')
    rho_air = get_density(temp_film, medium='air')
    nu_air = mu_air / rho_air  # kinematic viscocity

    # Reynolds number
    # for forced convection the characterisitc length is taken equal to the
    # module height
    L_forced = module_height
    Re = wind_speed * L_forced / nu_air

    # Nusselt number
    if Re <= 5 * 10**5:  # laminar flow
        Nu_forced = ((2 * 0.3387 * Re**0.5 * Pr_air**(1/3)) /
                     (1 + (0.0468 / Pr_air)**(2/3))**(1/4))
        # xxx: The original paper [1] erroneously uses an exponent of 1/5.
    else:  # turbulent flow
        Nu_forced = 2 * Pr_air**(1/3) * (0.037 * Re**(4/5) - 871)

    # Grashof number
    beta = 1 / C_to_K(temp_film)  # [1/K] thermal expansion coefficient
    g = 9.81  # [m/s^2]  gravity
    delta_t = (temp_film - temp_air)

    area = module_height * module_width
    perimeter = module_height * 2 + module_width * 2
    L_Gr = area / perimeter  # Characterisitc length for Grashof number
    Gr = g * beta * abs(delta_t) * L_Gr**3 / (nu_air**2)

    # Rayleigh number
    Ra = Gr * Pr_air

    Ra_cos = Ra * np.cos(np.deg2rad(module_tilt))

    # Modified to allow for negative delta_t (switches Nu case)
    if ((module_side == 'front') & (delta_t > 0)) | ((module_side == 'back') & ~(delta_t > 0)):
        if Ra_cos > 10**7:
            Nu_natural = 0.15 * Ra_cos**(1/3)
        else:
            Nu_natural = 0.54 * Ra_cos**(1/4)
    elif ((module_side == 'front') & ~(delta_t > 0)) | ((module_side == 'back') & (delta_t > 0)):
        Nu_natural = 0.27 * Ra_cos**(1/4)
    else:
        raise ValueError("Incorrect ``module_side`` provided, "
                         "has to be ``'front'`` or ``'back'``.")

    # Characterisitc length calculation
    volume = module_height * module_width * module_thickness
    L_free = volume / area

    # Calculate forced and natural convection heat transfer coefficients
    hc_forced = lamda_air * Nu_forced / L_forced
    hc_natural = lamda_air * Nu_natural / L_free
    # Overall heat transfer coefficient
    hc_total = (hc_forced**3 + hc_natural**3)**(1/3)
    return hc_total


def rahaman_analytical(temp_air, poa_global, wind_speed, temp_water,
                       module_height, module_width, module_thickness,
                       module_efficiency, module_tilt, temp_sky=None,
                       glass_cover_emissivity=0.91,
                       water_emissivity=0.95, backsheet_emissivity=0.85,
                       tau=0.95, alpha=0.95, unit_capacity=11000,
                       max_iterations=10, tol=0.01):
    """
    Estimate floating PV module temperature using the Rahaman simplest model.

    Rahaman's "simplest thermal model" [1]_ assumes the module's front, back,
    and cell temperature  are equal (i.e., no internal temperature gradient).
    It solves an energy balance  including convective and radiative heat
    transfer with both air and water, including electrical power extraction and
    thermal capacitance effects.

    Parameters
    ----------
    temp_air : pandas.Series or array-like
        Ambient air temperature [°C].
    temp_water : pandas.Series or array-like
        Water surface temperature underneath the PV modules [°C].
    poa_global : pandas.Series or array-like
        Plane‑of‑array global irradiance [W/m²].
    wind_speed : pandas.Series or array-like
        Wind speed at module height [m/s].
    module_height : float
        Module height (short dimension) [m].
    module_width : float
        Module width (long dimension) [m].
    module_thickness : float
        Module thickness (used for convective length scale) [m].
    module_efficiency : float
        Nominal electrical efficiency at 25 °C [-].
    module_tilt : float
        Tilt angle of the module relative to horizontal [°].
    temp_sky : numeric, optional
        Effective sky temperature for radiative loss [°C].
        Default: calculated from `temp_air`.
    glass_cover_emissivity : float, optional
        Emissivity of the front glass surface [-]. Default is 0.91.
    water_emissivity : float, optional
        Emissivity of the water surface [-]. Default is 0.95.
    backsheet_emissivity : float, optional
        Emissivity of the module backsheet [-]. Default is 0.85.
    tau : float, optional
        Transmittance of the front glass for shortwave radiation [-].
        Default is 0.95.
    alpha : float, optional
        Absorptance of the module for incident irradiance [-]. Default is 0.95.
    unit_capacity : float, optional
        Thermal capacity per unit area (heat capacity of module) [J/(m²·K)].
        Default is 11000.
    max_iterations : int, optional
        Maximum number of iterations per time step. Default is 10.
    tol : float, optional
        Convergence tolerance for temperature change per iteration [°C].
        Default is 0.01.

    Returns
    -------
    numpy.ndarray  
        Time series of module temperature estimates [°C].

    Notes
    -----
    - View factors for radiative exchange are assumed based on module tilt and geometry.
    - The model assumes the module has uniform temperature: T_cell = T_front = T_back.
    - Thermal lag (heat capacity) is included via an exponential decay term per time step.

    References
    ----------
    .. [1] M. A. Rahaman, T. L. Chambers, A. Fekih, G. Wiecheteck, G. Carranza,
           and G. R. C. Possetti, "Floating photovoltaic module temperature
           estimation: Modeling and comparison," Renewable Energy, vol. 208,
           pp. 162–180, Mar. 2023, :doi:`10.1016/j.renene.2023.03.076`.
    """
    timedelta_seconds = poa_global.index.to_series().diff().dt.total_seconds().copy()
    timedelta_seconds.iloc[0] = timedelta_seconds.iloc[1]

    sigma = 5.67e-8  # [W/m^2/K^4] Stefan–Boltzmann constant

    # Duffie & Beckmann claim view factor is 1 in this case
    vf_front = 0.5 * (1 + np.cos(np.deg2rad(module_tilt)))
    vf_back = 0.5 * (1 - np.cos(np.deg2rad(module_tilt)))
    back_emissivity = 1 / (1 / backsheet_emissivity + 1 / water_emissivity - 1)

    if temp_sky is None:
        temp_sky = 0.0522 * temp_air**1.5

    Tmod0 = np.nan  # initialize with dummy value
    Tmod_array = np.zeros_like(poa_global)

    reinitialize = True

    iterator = zip(temp_air, poa_global, wind_speed, temp_sky,
                   timedelta_seconds, temp_water)

    # iterate through timeseries inputs
    for i, (Tamb, sun, WS, Tsky, dtime, Twater) in enumerate(iterator):
        # solve the heat transfer equation, iterating because the heat loss
        # terms depend on tmod.

        # reinitilize values when encountering nan values
        if np.isnan(Tmod0):
            reinitialize = True
            Tmod0 = 20
            sun0 = sun

        Tmod = Tmod0

        Ps = tau * alpha * sun
        heatg = 0.002 * sun  # approximation glass absorptance

        for j in range(max_iterations):
            Pe = module_efficiency * Ps * (1 - 0.004 * (Tmod - 25))

            # convective coefficients
            h_c_f = _convection_coefficient(
                    Tamb, Tmod, WS, module_height, module_width,
                    module_thickness, module_tilt, module_side='front')
            h_c_b_1 = _convection_coefficient(
                    Tamb, Tmod, WS, module_height, module_width,
                    module_thickness, module_tilt, module_side='back')
            h_c_b_2 = 2.8 + 3 * WS
            h_c_b = (1 / h_c_b_1 + 1 / h_c_b_2)**-1
            
            # radiation coefficient sky
            h_r_f = vf_front * glass_cover_emissivity * sigma * (C_to_K(Tmod)**2 + C_to_K(Tsky)**2) * (C_to_K(Tmod) + C_to_K(Tsky))

            # radiation coefficient backside
            h_r_b = vf_back * back_emissivity * sigma * (C_to_K(Tmod)**2 + C_to_K(Twater)**2) * (C_to_K(Tmod) + C_to_K(Twater))

            # thermal lag
            L = - (h_c_f + h_c_b + h_r_f + h_r_b) *dtime / unit_capacity
            ex = np.exp(L)

            Tmod = Tmod0 * ex + (
                (1 - ex) * (
                    h_c_f * Tamb
                    + h_c_b * Twater
                    + h_r_f * Tsky
                    + h_r_b * Twater
                    - Pe
                    + heatg
                    + sun0 * tau * alpha
                    + (sun - sun0) * alpha / L
                ) + (sun - sun0) * tau * alpha
            ) / (h_c_f + h_c_b + h_r_f + h_r_b)

            # Convergence check
            if abs(Tmod - Tmod0) < tol:
                break

            # elimintate influence of thermal capacity when the prevous time
            # step contains nan values
            if reinitialize:
                Tmod0 = Tmod

        # Append and update values
        Tmod_array[i] = Tmod
        Tmod0 = Tmod
        sun0 = sun
        reinitialize = False  # reset reinitilization variable

    return Tmod_array
