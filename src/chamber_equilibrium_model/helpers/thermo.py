import aerosandbox.numpy as np
from proptools import constants
from cantera import Solution
import matplotlib.pyplot as plt
from functools import partial
from numpy.typing import NDArray


def make_stoich_coef_matrix(gas_obj):
    """Make a n_species by n_elements matrix dexcribing the number of atoms of
    element i per molecule species j.

    Args:
        gas_obj: cantera.Solution object
            representing combustion products; used only for looking up
            thermodynamic properties of combustion products

    Returns:
        n_species x n_elements matrix: describing the number of atoms of
            element i per molecule of species j
    """
    n_species = gas_obj.n_species
    n_elements = gas_obj.n_elements

    stoich_coef_mat = np.zeros((n_species, n_elements))
    for species_index in range(n_species):
        species_obj = gas_obj.species(species_index)
        species_comp = species_obj.composition
        for element in species_comp:
            element_index = gas_obj.element_index(element)
            element_val = species_comp[element]
            stoich_coef_mat[species_index, element_index] = element_val

    return stoich_coef_mat


def get_elements_in_propellant(
    prop_formula: dict,
    prop_ingredients_dict: dict,
) -> list:
    """Get a list of elements in propellant ingredients.

    Args:
        prop_formula_dict: a dictionary with keys that are
                propellant ingredients and values that are mass fractions,
                e.g. {"AP": 0.8, "HTPB+Curative": 0.2}
        prop_ingredients_dict: a dictionary describing the molecular weights,
            heats of formation, and chemical formula for possible propellant
            ingredients, see prop_ingredients.yaml

    Returns:
        a list of elements in the propellant / reaction
    """
    elements_in_prop = []

    for ingredient in prop_formula:
        ing_formula = prop_ingredients_dict["species"][ingredient]["formula"]
        for element in ing_formula:
            if element not in elements_in_prop:
                elements_in_prop.append(element)

    return elements_in_prop


def get_species_to_include(elements_in_prop: list, gas_obj: Solution) -> list:
    """Determine which species from the gas_obj to include in chamber
    equilibrium calculations.

    Args:
        elements_in_prop: list of chemical elements in propellant
        gas_obj: cantera.Solution object
            representing combustion products; used only for looking up
            thermodynamic properties of combustion products

    Returns:
        list of species names to include in equilibrium calculations
    """
    species_to_include = []
    for species_name in gas_obj.species_names:
        species_obj = gas_obj.species(species_name)
        species_comp = species_obj.composition

        # check if all elements in the species exist in the propellant formula
        if all(elem in elements_in_prop for elem in species_comp):
            species_to_include.append(species_name)

    return species_to_include


def enthalpy_func_one_piece(
    temperature: NDArray[np.float64],
    coefs: NDArray[np.float64],
    zone: int,
):
    """NASA 9-Coefficient standard enthalpy parameterization.

    Args:
        temperature (NDArray[np.float64]): gas temperatures to evaluate, [units: K]
        coefs (NDArray[np.float64]): NASA-9 coefficients for respective temperature
            (lookup table of coefficients in products.yaml)
        zone (int): index of temperature region for NASA-9 coefficients (which are
            often specified with different sets of coeffients for different
            temperature ranges)

    Returns:
        float: standard enthalpy at temperatures, [units: J mol**-1]
    """
    zone_shift = 11 * zone
    h = (
        temperature
        * constants.R_univ
        * (
            -coefs[3 + zone_shift] * temperature ** (-2)
            + coefs[4 + zone_shift] * np.log(temperature) / temperature
            + coefs[5 + zone_shift]
            + coefs[6 + zone_shift] / 2 * temperature
            + coefs[7 + zone_shift] / 3 * temperature**2
            + coefs[8 + zone_shift] / 4 * temperature**3
            + coefs[9 + zone_shift] / 5 * temperature**4
            + coefs[10 + zone_shift] / temperature
        )
    )

    return h


def enthalpy_func_blended(
    temperature: NDArray[np.float64],
    coefs: NDArray[np.float64],
):
    """Blended NASA 9-Coefficient enthalpy models. Use
        aerosandbox.numpy.blend() method to smoothly blend fits for different
        NASA-9 temperature ranges to satisfy C1-continuity requirement for
        aerosandbox solver.

    Args:
        temperature (NDArray[np.float64]): gas temperatures to evaluate,
            [units: K]
        coefs: NASA-9 coefficients, using cantera.Solution.thermo.coefs
            formatting

    Returns:
        the smoothed enthalpies over the specified temperatures
    """
    num_zones = int(coefs[0])

    h = enthalpy_func_one_piece(
        temperature=temperature,
        coefs=coefs,
        zone=0,
    )

    for zone in range(1, num_zones):
        zone_shift = 11 * zone
        temp_switch = coefs[1 + zone_shift]
        if temp_switch > 4000:  # no need for temperatures past 4000
            break

        h_zone = enthalpy_func_one_piece(
            temperature=temperature,
            coefs=coefs,
            zone=zone,
        )
        switch = 2e-2 * (temperature - temp_switch)
        h = np.blend(
            switch=switch,
            value_switch_high=h_zone,
            value_switch_low=h,
        )

    return h


def entropy_func_one_piece(
    temperature: NDArray[np.float64],
    coefs: NDArray[np.float64],
    zone: int,
):
    """NASA 9-Coefficient standard entropy parameterization.

    Args:
        temperature (NDArray[np.float64]): gas temperatures to evaluate, [units: K]
        coefs (NDArray[np.float64]): NASA-9 coefficients for respective temperature
            (lookup table of coefficients in products.yaml)
        zone (int): index of temperature region for NASA-9 coefficients (which are
            often specified with different sets of coeffients for different
            temperature ranges)

    Returns:
        float: standard enthalpy at temperatures, [units: J mol**-1]
    """
    zone_shift = 11 * zone
    s = constants.R_univ * (
        -coefs[3 + zone_shift] / 2 * temperature ** (-2)
        - coefs[4 + zone_shift] / temperature
        + coefs[5 + zone_shift] * np.log(temperature)
        + coefs[6 + zone_shift] * temperature
        + coefs[7 + zone_shift] / 2 * temperature**2
        + coefs[8 + zone_shift] / 3 * temperature**3
        + coefs[9 + zone_shift] / 4 * temperature**4
        + coefs[11 + zone_shift]
    )

    return s


def entropy_func_blended(
    temperature: NDArray[np.float64],
    coefs: NDArray[np.float64],
):
    """Blended NASA 9-Coefficient entropy models. Use
        aerosandbox.numpy.blend() method to smoothly blend fits for different
        NASA-9 temperature ranges to satisfy C1-continuity requirement for
        aerosandbox solver.

    Args:
        temperature (NDArray[np.float64]): gas temperatures to evaluate,
            [units: K]
        coefs: NASA-9 coefficients, using cantera.Solution.thermo.coefs
            formatting

    Returns:
        the smoothed entropies over the specified temperatures
    """
    num_zones = int(coefs[0])

    s = entropy_func_one_piece(
        temperature=temperature,
        coefs=coefs,
        zone=0,
    )

    for zone in range(1, num_zones):
        zone_shift = 11 * zone
        temp_switch = coefs[1 + zone_shift]
        if temp_switch > 4000:  # no need for temperatures past 4000
            break

        s_zone = entropy_func_one_piece(
            temperature=temperature,
            coefs=coefs,
            zone=zone,
        )
        switch = 2e-2 * (temperature - temp_switch)
        s = np.blend(
            switch=switch,
            value_switch_high=s_zone,
            value_switch_low=s,
        )

    return s


def heat_capacity_func_one_piece(
    temperature: NDArray[np.float64],
    coefs: NDArray[np.float64],
    zone: int,
):
    """NASA 9-Coefficient standard heat capcity at constant pressure
    parameterization.

    Args:
        temperature (NDArray[np.float64]): gas temperatures to evaluate, [units: K]
        coefs (NDArray[np.float64]): NASA-9 coefficients for respective temperature
            (lookup table of coefficients in products.yaml)
        zone (int): index of temperature region for NASA-9 coefficients (which are
            often specified with different sets of coeffients for different
            temperature ranges)

    Returns:
        standard heat capacity at temperatures, [units: J mol**-1 K**-1]
    """
    zone_shift = 11 * zone
    cp = constants.R_univ * (
        +coefs[3 + zone_shift] * temperature ** (-2)
        + coefs[4 + zone_shift] / temperature
        + coefs[5 + zone_shift]
        + coefs[6 + zone_shift] * temperature
        + coefs[7 + zone_shift] * temperature**2
        + coefs[8 + zone_shift] * temperature**3
        + coefs[9 + zone_shift] * temperature**4
    )

    return cp


def heat_capacity_func_blended(
    temperature: NDArray[np.float64],
    coefs: NDArray[np.float64],
):
    """Blended NASA 9-Coefficient heat capacity models. Use
        aerosandbox.numpy.blend() method to smoothly blend fits for different
        NASA-9 temperature ranges to satisfy C1-continuity requirement for
        aerosandbox solver.

    Args:
        temperature (NDArray[np.float64]): gas temperatures to evaluate,
            [units: K]
        coefs: NASA-9 coefficients, using cantera.Solution.thermo.coefs
            formatting

    Returns:
        the smoothed heat capacities at constant pressure over the specified
            temperatures
    """
    num_zones = int(coefs[0])

    cp = heat_capacity_func_one_piece(
        temperature=temperature,
        coefs=coefs,
        zone=0,
    )

    for zone in range(1, num_zones):
        zone_shift = 11 * zone
        temp_switch = coefs[1 + zone_shift]
        if temp_switch > 4000:  # no need for temperatures past 4000
            break

        cp_zone = heat_capacity_func_one_piece(
            temperature=temperature,
            coefs=coefs,
            zone=zone,
        )
        switch = 2e-2 * (temperature - temp_switch)
        cp = np.blend(
            switch=switch,
            value_switch_high=cp_zone,
            value_switch_low=cp,
        )

    return cp


def make_thermo_funcs(
    gas_obj: Solution,
):
    """Return a dictionary containing three lists of functions, describing
    entropy and enthalpy for each species.

    Args:
        gas_obj: cantera.Solution object
            representing combustion products; used only for looking up
            thermodynamic properties of combustion products

    Returns:
        dictionary containining functions for entropy, enthalpy, and heat
            capacity for all species
    """
    species_names = gas_obj.species_names

    enthalpy_func_list = []
    entropy_func_list = []
    cp_func_list = []

    for species in species_names:
        species_obj = gas_obj.species(species)
        coefs = species_obj.thermo.coeffs

        enthalpy_func_list.append(partial(enthalpy_func_blended, coefs=coefs))
        entropy_func_list.append(partial(entropy_func_blended, coefs=coefs))
        cp_func_list.append(partial(heat_capacity_func_blended, coefs=coefs))

    out = {
        "entropy_funcs": entropy_func_list,
        "enthalpy_funcs": enthalpy_func_list,
        "heat_cap_funcs": cp_func_list,
    }

    return out


def main():
    yaml_file = "products.yaml"

    gas_obj = Solution(yaml_file)

    thermo_func_dict = make_thermo_funcs(gas_obj=gas_obj)
    entropy_funcs = thermo_func_dict["entropy_funcs"]
    enthalpy_funcs = thermo_func_dict["enthalpy_funcs"]
    heat_cap_funcs = thermo_func_dict["heat_cap_funcs"]

    temps_test = np.linspace(298, 6000, 500)

    species_names = gas_obj.species_names

    for index, name in enumerate(species_names):
        species_obj = gas_obj.species(name)

        h_test = np.array([species_obj.thermo.h(temp) for temp in temps_test]) * 1e-3  # type: ignore[reportOperatorIssue]
        s_test = np.array([species_obj.thermo.s(temp) for temp in temps_test]) * 1e-3  # type: ignore[reportOperatorIssue]
        cp_test = np.array([species_obj.thermo.cp(temp) for temp in temps_test]) * 1e-3  # type: ignore[reportOperatorIssue]

        g_test = []
        for temp in temps_test:
            gas_obj.TP = temp, 101325
            g_test.append(gas_obj.standard_gibbs_RT[index] * constants.R_univ * temp)

        h = enthalpy_funcs[index](temps_test)
        s = entropy_funcs[index](temps_test)
        g = h - temps_test * s
        cp = heat_cap_funcs[index](temps_test)

        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(14, 7)
        fig.suptitle("Surrogate Models for {species}".format(species=name))

        axs[0, 0].set_title("Enthalpy")
        axs[0, 0].set_xlabel("Temperature, [K]")
        axs[0, 0].set_ylabel("Enthalpy, [J mol**-1]")
        axs[0, 0].plot(temps_test, h_test, label="Cantera output")
        axs[0, 0].plot(temps_test, h, "--", label="Surrogate model output")
        axs[0, 0].legend()

        axs[0, 1].set_title("Entropy")
        axs[0, 1].set_xlabel("Temperature, [K]")
        axs[0, 1].set_ylabel("Entropy, [J mol**-1 K**-1]")
        axs[0, 1].plot(temps_test, s_test, label="Cantera output")
        axs[0, 1].plot(temps_test, s, "--", label="Surrogate model output")
        axs[0, 1].legend()

        axs[1, 0].set_title("Gibb's Free Energy")
        axs[1, 0].set_xlabel("Temperature, [K]")
        axs[1, 0].set_ylabel("Entropy, [J mol**-1]")
        axs[1, 0].plot(temps_test, g_test, label="Cantera output")
        axs[1, 0].plot(temps_test, g, "--", label="Surrogate model output")
        axs[1, 0].legend()

        axs[1, 1].set_title("Heat Capacity at Constant Pressure")
        axs[1, 1].set_xlabel("Temperature, [K]")
        axs[1, 1].set_ylabel("Heat Capacity, [J mol**-1 K**-1]")
        axs[1, 1].plot(temps_test, cp_test, label="Cantera output")
        axs[1, 1].plot(temps_test, cp, "--", label="Surrogate model output")
        axs[1, 1].legend()

        plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
