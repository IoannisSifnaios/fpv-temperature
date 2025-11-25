"""FPV temperature model from Lindholm."""

import numpy as np
from properties import (get_thermal_conductivity, get_prandtl,
                        get_dynamic_viscosity, get_density,
                        get_cp, C_to_K)


def _convection_coefficient(
        temp_air, temp_surface, wind_speed, module_width, module_height,
        module_side, convection_method='cegel'):
    # Convective heat transfer coefficients are eval. at the film temp.
    temp_film = (temp_air + temp_surface) / 2

    # Characterisitc length calculation for natural and forced convection
    L_natural = (module_width * module_height /
                 (2 * (module_width + module_height)))

    L_forced = (4 * module_width * module_height /
                (2 * (module_width + module_height)))

    # Air properties
    lamda_air = get_thermal_conductivity(temp_film, medium='air')
    Pr_air = get_prandtl(temp_film, medium='air')
    mu_air = get_dynamic_viscosity(temp_film, medium='air')
    rho_air = get_density(temp_film, medium='air')
    nu_air = mu_air / rho_air  # kinematic viscocity

    # Reynolds number
    Re = wind_speed * L_forced / nu_air

    # Nusselt number
    if Re < 5 * 10**5:  # laminar flow
        Nu_forced = 0.664 * Re**(1/2) * Pr_air**(1/3)
    else:  # turbulent flow
        Nu_forced = (0.037 * Re**(4/5) - 871) * Pr_air**(1/3)

    beta = 1 / (temp_film + 273.15)  # [1/K] thermal expansion coefficient
    g = 9.81  # [m/s^2]  gravity
    specific_heat = get_cp(temp_film, medium='air')  # [J/kg/K]

    # Rayleigh number
    delta_t = (temp_film - temp_air)
    Ra = ((g * beta * abs(delta_t) * L_natural**3) /
          (nu_air * lamda_air / (rho_air * specific_heat)))

    if convection_method not in ['bergman', 'cegel']:
        raise ValueError("Incorrect ``nusselt_method`` provided, "
                         "has to be ``'bergman'`` or ``'cegel'``.")

    # Modified to allow for negative delta_t (switches Nu case)
    if ((module_side == 'front') & ~(delta_t < 0)) | ((module_side == 'back') & (delta_t < 0)):
        if convection_method == 'bergman':
            Nu_natural = 0.15 * Ra**(1/3)
        elif convection_method == 'cegel':
            if Ra > 10**7:
                Nu_natural = 0.10 * Ra**(1/3)
            else:
                Nu_natural = 0.59 * Ra**(1/4)
    elif ((module_side == 'front') & (delta_t < 0)) | ((module_side == 'back') & ~(delta_t < 0)):
        if convection_method == 'bergman':
            Nu_natural = 0.52 * Ra**(1/5)
        elif convection_method == 'cegel':
            Nu_natural = 0.27 * Ra**(1/4)
    else:
        raise ValueError("Incorrect ``module_side`` provided, "
                         "has to be ``'front'`` or ``'back'``.")

    # Calculate forced and natural convection heat transfer coefficients
    hc_forced = lamda_air * Nu_forced / L_forced
    hc_natural = lamda_air * Nu_natural / L_natural
    # Overall heat transfer coefficient
    hc_total = (hc_forced**3 + hc_natural**3)**(1/3)
    return hc_total


def lindholm(poa_global, temp_air, wind_speed, temp_water,
             module_height, module_width, module_efficiency, temp_sky=None,
             module_temp_coefficient=-0.004, convection_method='cegel',
             sky_emissivity=0.8, glass_cover_emissivity=0.88,
             backsheet_emissivity=0.9, water_emissivity=0.94,
             alpha_absorption=0.9, glass_transmittance=1,
             glass_thickness_front=0.0025, backsheet_thickness=0.0003,
             encapsulant_thickness_front=0.0004, encapsulant_thickness_back=0.0004,
             wafer_thickness=0.000175, glass_conductivity_front=1.8,
             backsheet_conductivity=0.2, encapsulant_conductivity_front=0.21,
             encapsulant_conductivity_back=0.21, wafer_conductivity=148,
             max_iterations=10, tol=0.01):
    r"""
    Estimate cell temperature for FPV system using the Lindholm model.

    The Lindholm model [1]_ solves the coupled front- and back-surface heat
    balance equations for a PV module installed over water. The model accounts
    for:

      • absorbed solar irradiance
      • module electrical conversion efficiency
      • temperature-dependent efficiency losses
      • conductive heat transfer through module layers
      • convective heat losses on front and back surfaces
      • radiative exchange with the sky (front) and water surface (back)
      • forced + natural convection computed via several correlations
      • iterative solution of the nonlinear energy balance

    The solution is obtained via fixed-point iteration until front and back
    temperatures converge within a specified tolerance.

    Parameters
    ----------
    poa_global : array-like
        Plane-of-array global irradiance incident on the module [W/m²].
    temp_air : array-like
        Ambient air temperature [°C].
    wind_speed : array-like
        Wind speed at module height [m/s].
    temp_water : array-like
        Water surface temperature beneath the module [°C].
    module_height : float
        Physical module height (short dimension) [m].
    module_width : float
        Physical module width (long dimension) [m].
    module_efficiency : float
        Nominal power conversion efficiency at 25°C [-].
    temp_sky : array-like, optional
        Effective sky temperature used for radiative heat loss [°C]. If
        ``None``, the model estimates it using the Swinbank relationship
        :math:`T_{sky} = 0.0552 \, T_{air}^{1.5}`.
    module_temp_coefficient : float, optional
        Temperature coefficient of module efficiency [1/°C]. Default is -0.004.
    convection_method : {'cegel', 'bergman', 'watmuff', 'mcadams'}, optional
        Method used for estimating convective heat-transfer coefficients.
        ``'cegel'`` and ``'bergman'`` compute forced + natural convection using
        air properties and Nusselt correlations. ``'watmuff'`` and
        ``'mcadams'`` use simplified empirical linear models.  
        Default is ``'cegel'``.
    sky_emissivity : float, optional
        Effective emissivity of the sky [-]. Default is 0.8.
    glass_cover_emissivity : float, optional
        Emissivity of the glass cover [-]. Default is 0.88.
    backsheet_emissivity : float, optional
        Emissivity of the backsheet surface [-]. Default is 0.9.
    water_emissivity : float, optional
        Emissivity of the water surface [-]. Default is 0.94.
    alpha_absorption : float, optional
        Thermal absorptance of the module [-]. Default is 0.9.
    glass_transmittance : float, optional
        Transmittance of the glass in the solar spectrum [-]. Default is 1.0.
    glass_thickness_front : float, optional
        Front glass thickness [m]. Default is 0.0025.
    backsheet_thickness : float, optional
        Backsheet thickness [m]. Default is 0.0003.
    encapsulant_thickness_front : float, optional
        Front encapsulant thickness [m]. Default is 0.0004.
    encapsulant_thickness_back : float, optional
        Back encapsulant thickness [m]. Default is 0.0004.
    wafer_thickness : float, optional
        Silicon wafer thickness [m]. Default is 0.000175.
    glass_conductivity_front : float, optional
        Thermal conductivity of front glass [W/m/K]. Default is 1.8.
    backsheet_conductivity : float, optional
        Thermal conductivity of backsheet [W/m/K]. Default is 0.2.
    encapsulant_conductivity_front : float, optional
        Thermal conductivity of front encapsulant [W/m/K]. Default is 0.21.
    encapsulant_conductivity_back : float, optional
        Thermal conductivity of back encapsulant [W/m/K]. Default is 0.21.
    wafer_conductivity : float, optional
        Thermal conductivity of silicon wafer [W/m/K]. Default is 148.
    max_iterations : int, optional
        Maximum number of iterations for calculating module temperatures [-].
        Default is 10.
    tol : float, optional
        Convergence tolerance for module temperatures [°C]. Default is 0.01.

    Returns
    -------
    numpy.ndarray
        Array of estimated cell temperatures in °C.

    Raises
    ------
    ValueError
        If an unsupported convection method is specified.

    Notes
    -----
    The model solves the following coupled nonlinear system:

    - Front-surface heat balance  
    - Back-surface heat balance  
    - Cell energy balance including electrical power extraction  

    The characteristic length for forced convection is the hydraulic diameter
    of the module:

    :math:`L = \\frac{4HW}{2(H+W)}`

    where *H* = module height and *W* = module width.

    For natural convection, the characterisitc length is taken equal to:

    :math:`L = \\frac{HW}{2(H+W)}`

    Convective heat transfer on each surface is computed using the specified
    Nusselt correlation and air properties evaluated at the film temperature.

    References
    ----------
    .. [1] D. Lindholm, T. Kjeldstad, J. Selj, E. Stensrud Marstein, and H. G.
           Fjær, "Heat loss coefficients computed for floating PV modules,"
           Progress in Photovoltaics: Research and Applications, vol. 29, no.
           12, pp. 1262–1273, Dec. 2021. :doi:`10.1002/pip.3451`.
    """
    emissivity_front = 1 / (
        1 / glass_cover_emissivity + 1 / sky_emissivity - 1)

    emissivity_back = 1 / (
        1 / backsheet_emissivity + 1 / water_emissivity - 1)

    # Thermal resistance front
    A_front = 1 / (
        glass_thickness_front / glass_conductivity_front +
        encapsulant_thickness_front / encapsulant_conductivity_front +
        (wafer_thickness / 2) / wafer_conductivity
    )

    # Thermal resistance back
    A_back = 1 / (
        backsheet_thickness / backsheet_conductivity +
        encapsulant_thickness_back / encapsulant_conductivity_back +
        (wafer_thickness / 2) / wafer_conductivity
    )

    # Estiamte sky temperature if not provided
    if temp_sky is None:
        temp_sky = 0.0552 * temp_air**1.5

    sigma = 5.67e-8  # [W/m^2/K^4] Stefan–Boltzmann constant

    TC_return = []

    # Allowing for inputs to be arrays or series
    poa_globals, temp_airs, wind_speeds, temp_waters, temp_skys = \
        poa_global, temp_air, wind_speed, temp_water, temp_sky

    for poa_global, temp_air, wind_speed, temp_water, temp_sky in zip(
            poa_globals, temp_airs, wind_speeds, temp_waters, temp_skys):

        # Initial guess
        TC = temp_air + 5
        T_front = TC
        T_back = TC

        # Iteration loop
        for _ in range(max_iterations):
            if convection_method == 'watmuff':
                hc = 2.8 + 3.0 * wind_speed
                hc_front, hc_back = hc, hc
            elif convection_method == 'mcadams':
                hc = 5.7 + 3.8 * wind_speed
                hc_front, hc_back = hc, hc
            elif convection_method in ['cegel', 'bergman']:
                hc_front = _convection_coefficient(
                    temp_air, T_front, wind_speed, module_width, module_height,
                    'front', convection_method=convection_method)
                hc_back = _convection_coefficient(
                    temp_air, T_back, wind_speed, module_width, module_height,
                    'back', convection_method=convection_method)
            else:
                raise ValueError("The specified ``convection_method`` is not supported.")

            h_front_rad = emissivity_front * sigma * (
                (C_to_K(T_front) + C_to_K(temp_sky)) *
                (C_to_K(T_front)**2 + C_to_K(temp_sky)**2))

            h_back_rad = emissivity_back * sigma * (
                (C_to_K(T_back) + C_to_K(temp_water)) *
                (C_to_K(T_back)**2 + C_to_K(temp_water)**2))

            B_front = A_front + hc_front + h_front_rad
            C_front = hc_front * temp_air + h_front_rad * temp_sky

            B_back = A_back + hc_back + h_back_rad
            C_back = hc_back * temp_air + h_back_rad * temp_water

            module_eff_mod = (module_efficiency *
                              (1 + module_temp_coefficient * (TC - 25)))

            # Compute new TC
            TC_new = (
                (B_front * B_back * (
                    alpha_absorption * glass_transmittance - module_eff_mod) * poa_global +
                 A_front * B_back * C_front + A_back * B_front * C_back) /
                (B_front * B_back * (A_back + A_front) - A_back**2 * B_front - A_front**2 * B_back)
            )

            # Update front and back temperatures
            T_front_new = (A_front * TC_new + C_front) / B_front
            T_back_new = (A_back * TC_new + C_back) / B_back

            # Convergence check
            if abs(T_front_new - T_front) < tol and abs(T_back_new - T_back) < tol:
                break

            # Update for next iteration
            TC, T_front, T_back = TC_new, T_front_new, T_back_new

        TC_return.append(TC_new)

    return np.array(TC_return)
