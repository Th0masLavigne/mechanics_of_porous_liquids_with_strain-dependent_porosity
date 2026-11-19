# Date: 09/09/2025
# Author: Thomas Lavigne
# Reference: Giuseppe Sciume
# Laboratory: I2M
#
# Update: 20/07/2026
# Added Linear_pressure_law and Interfacial_curvature, introduced together with
# the corrected formulation from the reference model porous_liquid_2026_all3.py
# (see change log delivered with this correction).
#
import functions
import dolfinx
#
# --------------------------------------- #
# 			  Constitutive laws			  #
# --------------------------------------- #
#
def Byrne_coefficients(equilibrium_fraction:dolfinx.default_scalar_type,interaction_fraction:dolfinx.default_scalar_type,decohesion_fraction:dolfinx.default_scalar_type,decohesion_pressure:dolfinx.default_scalar_type,**kwargs):
    """
    Calculates the material-specific coefficients (alpha and beta) for the
    Byrne cohesive zone model constitutive law.

    These coefficients are derived from four key material-dependent fractions
    and the characteristic decohesion pressure, ensuring the cohesive traction
    function has the desired shape.

    Parameters
    ----------
    equilibrium_fraction : float
        The damage fraction (s) at which the cohesive traction returns to zero (epsilon, $\varepsilon$).
    interaction_fraction : float
        The damage fraction (s) at which the interaction/decohesion process begins (eta, $\eta$).
    decohesion_fraction : float
        The damage fraction (s) defining a characteristic point of the material response (delta, $\delta$).
    decohesion_pressure : float
        The characteristic pressure magnitude (Pd, typically negative for tensile stress).
    **kwargs : dict, optional
        Optional keyword arguments, currently only supports `log` for printing coefficients.

    Returns
    -------
    tuple[float, float]
        The calculated coefficients (alpha, beta).

    Raises
    ------
    ValueError
        If the calculated coefficient `beta` is negative, indicating inconsistent input fractions.
    """
    import dolfinx
    import ufl
    log = kwargs.get('log', None)
    #
    beta   = dolfinx.default_scalar_type( ((1 - decohesion_fraction) * (3 * decohesion_fraction - 2 * equilibrium_fraction - interaction_fraction)) / ((decohesion_fraction - interaction_fraction) * (equilibrium_fraction - decohesion_fraction)) )
    if beta<0:
        raise ValueError(f'The coefficient beta={beta} has turned negative. Please adjust the fraction values.')
    alpha  = dolfinx.default_scalar_type((-decohesion_pressure * (1 - decohesion_fraction)**beta) / ((decohesion_fraction - interaction_fraction)**2 * (equilibrium_fraction - decohesion_fraction)))
    #
    import mpi4py
    if log:
        if mpi4py.MPI.COMM_WORLD.rank == 0:
            print(f"""
                The Byrne coefficients are :
                    - alpha = {alpha}
                    - beta  = {beta}
                """)
    return alpha,beta
#
def Byrne_law(porosity:dolfinx.fem.Function,equilibrium_fraction:dolfinx.default_scalar_type,interaction_fraction:dolfinx.default_scalar_type,decohesion_fraction:dolfinx.default_scalar_type,decohesion_pressure:dolfinx.default_scalar_type,**kwargs):
    """
    Implements the full non-linear Byrne constitutive law for cohesive traction
    as a UFL expression.

    The cohesive pressure (traction) P is a function of the damage variable s (`fraction_s`).
    The law is zero until the damage fraction exceeds the interaction fraction (eta, $\eta$).

    $$ P(s) = \begin{cases} \alpha \frac{(s - \eta)^2 (s - \varepsilon)}{(1 - s)^\beta} & \text{if } s > \eta \\ 0 & \text{if } s \le \eta \end{cases} $$

    [Image of cohesive traction-separation curve]

    Parameters
    ----------
    fraction_s : dolfinx.fem.Function
        The damage fraction field (s).
    equilibrium_fraction : float
        The fraction $\varepsilon$.
    interaction_fraction : float
        The fraction $\eta$.
    decohesion_fraction : float
        The fraction $\delta$.
    decohesion_pressure : float
        The characteristic pressure $P_d$.
    **kwargs : dict, optional
        Optional keyword arguments, currently only supports `log`.

    Returns
    -------
    ufl.conditional
        A UFL expression representing the cohesive pressure P(s).
    """
    import dolfinx
    import ufl
    log = kwargs.get('log', None)
    if log :
        alpha, beta = Byrne_coefficients(equilibrium_fraction,interaction_fraction,decohesion_fraction,decohesion_pressure,log=True)
    else :
        alpha, beta = Byrne_coefficients(equilibrium_fraction,interaction_fraction,decohesion_fraction,decohesion_pressure)
    # condition, true value, false value
    return ufl.conditional(ufl.lt(interaction_fraction, porosity), alpha * ((porosity - interaction_fraction)**2 * (porosity - equilibrium_fraction)) / (1 - porosity)**beta, 0)
#
def Linearized_Byrne(porosity:dolfinx.fem.Function,equilibrium_fraction:dolfinx.default_scalar_type,interaction_fraction:dolfinx.default_scalar_type,decohesion_fraction:dolfinx.default_scalar_type,decohesion_pressure:dolfinx.default_scalar_type,**kwargs):
    """
    Implements a simplified, linearized form of the Byrne constitutive law
    around the equilibrium fraction $\varepsilon$.

    This linearized form is often used to define the tangent stiffness (or Jacobian)
    in Newton-Raphson iterative solvers for numerical stability and convergence.

    The linearization is given by $P_{lin}(s) = B(s - \varepsilon)$, where $B$ is a
    derived coefficient (tangent modulus approximation) at $s=\varepsilon$.

    Parameters
    ----------
    fraction_s : dolfinx.fem.Function
        The damage fraction field (s).
    equilibrium_fraction : float
        The fraction $\varepsilon$.
    interaction_fraction : float
        The fraction $\eta$.
    decohesion_fraction : float
        The fraction $\delta$.
    decohesion_pressure : float
        The characteristic pressure $P_d$.
    **kwargs : dict, optional
        Optional keyword arguments, currently only supports `log`.

    Returns
    -------
    ufl.Product
        A UFL expression representing the linearized cohesive pressure.
    """
    import dolfinx
    import ufl
    log = kwargs.get('log', None)
    if log :
        alpha, beta = Byrne_coefficients(equilibrium_fraction,interaction_fraction,decohesion_fraction,decohesion_pressure,log=True)
    else :
        alpha, beta = Byrne_coefficients(equilibrium_fraction,interaction_fraction,decohesion_fraction,decohesion_pressure)
    # ca fonctionnait avec le moins mais c'etait une erreur
    # return - alpha * (equilibrium_fraction-interaction_fraction)**2/((1-equilibrium_fraction)**beta)*(porosity-equilibrium_fraction)
    return alpha * (equilibrium_fraction-interaction_fraction)**2/((1-equilibrium_fraction)**beta)*(porosity-equilibrium_fraction)
#
def Linear_pressure_law(cell_fraction, equilibrium_fraction:dolfinx.default_scalar_type, bulk_modulus:dolfinx.default_scalar_type, **kwargs):
    """
    Linear (poro-elastic) constitutive law for the solid/scaffold pressure.

    This closure law replaces the non-linear Byrne cohesive law for models
    following the corrected formulation introduced in the reference legacy
    model ``porous_liquid_2026_all3.py``:

        p_s(epsilon^s) = (K_s / epsilon^s_0) * (epsilon^s - epsilon^s_0)

    Parameters
    ----------
    cell_fraction : ufl expression or dolfinx.fem.Function
        The current solid (cell) volume fraction field (epsilon^s).
    equilibrium_fraction : float
        The reference/equilibrium solid volume fraction (epsilon^s_0).
    bulk_modulus : float
        The bulk compressibility modulus of the solid/scaffold phase (K_s) [Pa].
    **kwargs : dict, optional
        Optional keyword arguments, currently only supports `log`, which prints
        the effective linearised (constrained) modulus dp_s/d(epsilon^s).

    Returns
    -------
    ufl.Expression
        The linear solid pressure p_s.
    """
    log = kwargs.get('log', None)
    if log:
        import mpi4py
        if mpi4py.MPI.COMM_WORLD.rank == 0:
            print(f"Linear solid pressure law: dp_s/d(eps_s) = K_s/eps_s0 = {bulk_modulus/equilibrium_fraction}")
    return (bulk_modulus/equilibrium_fraction) * (cell_fraction - equilibrium_fraction)
#
def Interfacial_curvature(cell_fraction, interfacial_density, equilibrium_fraction:dolfinx.default_scalar_type, interfacial_density_0:dolfinx.default_scalar_type, curvature_reference:dolfinx.default_scalar_type, curvature_solid_coefficient:dolfinx.default_scalar_type, curvature_interfacial_coefficient:dolfinx.default_scalar_type=0., **kwargs):
    """
    Computes the mean interfacial curvature J^sf of the solid-fluid interface,
    as a linearised function of the solid volume fraction and of the specific
    interfacial density itself.

        J^sf = J^sf_0 * ( 1 + A_s * (epsilon^s - epsilon^s_0) + A_sf * (epsilon^sf - epsilon^sf_0) )

    This closure law is introduced with the corrected formulation
    (``porous_liquid_2026_all3.py``) and feeds the extra Laplace (interfacial
    tension) pressure contribution gamma_sf * J^sf in the momentum balance,
    together with the transport equation of epsilon^sf.

    Parameters
    ----------
    cell_fraction : ufl expression or dolfinx.fem.Function
        The current solid (cell) volume fraction field (epsilon^s).
    interfacial_density : ufl expression or dolfinx.fem.Function
        The current specific interfacial density field (epsilon^sf).
    equilibrium_fraction : float
        The reference/equilibrium solid volume fraction (epsilon^s_0).
    interfacial_density_0 : float
        The reference specific interfacial density (epsilon^sf_0).
    curvature_reference : float
        The reference curvature J^sf_0 (computed from the particle radius).
    curvature_solid_coefficient : float
        The sensitivity coefficient A_s of the curvature to the solid fraction.
    curvature_interfacial_coefficient : float, optional
        The sensitivity coefficient A_sf of the curvature to the interfacial
        density itself (default 0., as in the reference formulation).
    **kwargs : dict, optional
        Optional keyword arguments, currently only supports `log`.

    Returns
    -------
    ufl.Expression
        The mean interfacial curvature J^sf.
    """
    log = kwargs.get('log', None)
    if log:
        import mpi4py
        if mpi4py.MPI.COMM_WORLD.rank == 0:
            print(f"Interfacial curvature law: J_sf0={curvature_reference}, As={curvature_solid_coefficient}, Asf={curvature_interfacial_coefficient}")
    return curvature_reference * ( 1+ curvature_solid_coefficient*(cell_fraction-equilibrium_fraction) + curvature_interfacial_coefficient*(interfacial_density-interfacial_density_0))
#
#____________________________________________________________#
#                   End of functions
#____________________________________________________________#
#
#
if __name__ == "__main__":
    print("Loading of the user-defined constitutive laws successfully completed.")
    # EoF
