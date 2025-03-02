import aerosandbox.numpy as np
import aerosandbox as asb
from aerosandbox.common import ImplicitAnalysis
from cantera import Solution
from pathlib import Path
import yaml
import json
from chamber_equilibrium_model.helpers import thermo
from proptools import constants
from typing import Optional
import casadi as cas
import matplotlib.pyplot as plt

# constants
P_STP = 101325  # Pa
R_UNIV = constants.R_univ  # J/(mol*K); universal gas constant


class ChamberEquilibrium(ImplicitAnalysis):
    @ImplicitAnalysis.initialize
    def __init__(
        self,
        propellant_formula: dict,
        p_c: float,
        p_e: float = 101325,
        T_c_guess=2200,
        T_exit_guess=1000,
        gas_obj_products: Optional[Solution] = None,
    ):
        """An implicit analysis for the internal ballistics of a solid rocket
        motor. Takes as inputs the propellant formula, chamber and
        exit chamber pressures, and optionally guesses for chamber and exit
        temperatures, as well as a Cantera.Solution object represent the
        combustion products of the propellant.

        The class implicitly makes a call to the AeroSandbox.opti.solve method
        to find a self-consistent solution for the equilibrium chamber
        temperature and species mole fractions (as well as all intermediate
        variables defined in the constraints below).

        Args:
            propellant_formula (dict): a dictionary with keys that are
                propellant ingredients and values that are mass fractions,
                e.g. {"AP": 0.8, "HTPB+Curative": 0.2}
            p_c (float): motor chamber pressure, [Pa]
            p_e (float): nozzle exit pressure, [Pa]
            gas_obj_products (cantera.Solution): cantera.Solution object
                representing combustion products; used only for looking up
                thermodynamic properties of combustion products
            T_c_guess (float): guess at the chamber temperature, [K]
            T_exit_guess (float): guess at the nozzle exit temperature, [K]
        """

        self.prop_formula = propellant_formula
        self.p_c = p_c
        self.p_e = p_e
        self.T_c_guess = T_c_guess
        self.T_e_guess = T_exit_guess
        self.parent_directory = Path(__file__).parent.absolute()

        # make Cantera gas object, assign attribute
        if gas_obj_products is None:
            mechanism_path = self.parent_directory / "helpers/products.yaml"
            self.gas_obj = Solution(mechanism_path)
        else:
            self.gas_obj = gas_obj_products

        self._process_prop_formula()
        self._process_gas_species()

        self._setup_variables()
        self._enforce_governing_equations()
        self._process_results()

    def _setup_variables(self):
        # initial guesses
        n_moles_gas_guess = 1 / 0.025  # (1kg) / (a reasonable molecular weight)
        mol_chamber_guess = 1e-2 * np.ones(np.length(self.selected_species))
        mol_chamber_guess[self.selected_species.index("CO2")] = 0.1
        mol_chamber_guess[self.selected_species.index("CO")] = 0.1
        mol_chamber_guess[self.selected_species.index("H2O")] = 0.4
        mol_chamber_guess[self.selected_species.index("N2")] = 0.1
        mol_chamber_guess[self.selected_species.index("HCl")] = 0.2
        mol_chamber_guess = mol_chamber_guess * n_moles_gas_guess

        if "Al" in self.prop_formula:  # guess if all Al becomes Al2O3
            mol_Al_guess = self.prop_formula["Al"] / 0.02698 * 0.5
            mol_chamber_guess[self.selected_species.index("Al2O3(b)")] = mol_Al_guess

        # initialize chamber variables
        self.n_moles_gas_chamber = self.opti.variable(  # type: ignore[reportArgumentType]
            init_guess=n_moles_gas_guess, log_transform=True
        )
        self.mol_chamber = self.opti.variable(  # type: ignore[reportArgumentType]
            init_guess=mol_chamber_guess, log_transform=True
        )
        self.lagrange_mults_div_RT_chamber = self.opti.variable(  # type: ignore[reportArgumentType]
            init_guess=np.zeros(len(self.prop_elements))
        )
        self.temp_chamber = self.opti.variable(init_guess=self.T_c_guess)  # type: ignore[reportArgumentType]
        self.temp_exit = self.opti.variable(init_guess=self.T_e_guess)  # type: ignore[reportArgumentType]

    def _molar_entropy(
        self,
        temperature: float | cas.MX,
        pressure: float | cas.MX,
    ):
        """Calculate specific molar entropy of combustion products at different
        temperatures and pressures, assuming frozen flow.

        Args:
            temperature: station temperature, [K]
            pressure: station pressure, [Pa]

        Returns:
            gas molar entropy, [J mol**-1 K**-1]
        """

        molar_entropies = np.array(  # J / mol K
            [func(temperature) for func in self.entropy_funcs]
        )
        molar_entropy_gas_adjustment = -R_UNIV * np.log(
            self.mol_chamber[self.index_gas] / self.n_moles_gas_chamber
        ) - R_UNIV * np.log(pressure / P_STP)
        molar_entropies[self.index_gas] += molar_entropy_gas_adjustment  # type: ignore[reportArgumentType]
        adjusted_molar_entropy = np.sum(molar_entropies * self.mol_chamber)  # type: ignore[reportArgumentType]

        return adjusted_molar_entropy

    def _enforce_governing_equations(self):
        """Enforce thermodynamic equilibrium governing equations."""

        # get product species enthalpies, entropies, and gibbs free energies
        species_standard_enthalpies = np.array(  # J / mol
            [func(self.temp_chamber) for func in self.enthalpy_funcs]
        )

        species_standard_entropies = np.array(  # J / mol K
            [func(self.temp_chamber) for func in self.entropy_funcs]
        )
        species_gibbs_energies = (  # J / mol
            species_standard_enthalpies - self.temp_chamber * species_standard_entropies
        )

        # minimize Gibb's free energy for gaseous species
        self.opti.subject_to(
            species_gibbs_energies[self.index_gas] / (R_UNIV * self.temp_chamber)
            + np.log(self.mol_chamber[self.index_gas] / self.n_moles_gas_chamber)
            + np.log(self.p_c / P_STP)
            - (
                self.prod_stoich_coef_mat[self.index_gas, :]
                @ self.lagrange_mults_div_RT_chamber
            )
            == 0
        )

        # minimize Gibb's free energy for condensed phase species
        self.opti.subject_to(
            species_gibbs_energies[self.index_condensed] / (R_UNIV * self.temp_chamber)
            - (
                self.prod_stoich_coef_mat[self.index_condensed, :]
                @ self.lagrange_mults_div_RT_chamber
            )
            == 0
        )

        # conservation of mass for each species, arbitrarily assuming 1 kg
        # system mass
        self.opti.subject_to(
            (
                (self.prod_stoich_coef_mat.T @ self.mol_chamber)
                - (
                    self.reac_stoich_coef_mat.T
                    @ (self.mass_fracs_ing / self.mol_weight_ing)  # type: ignore[reportOperatorIssue]
                )
            )
            == 0
        )

        # sum of gas moles equals n mol total gas
        self.opti.subject_to(
            np.sum(self.mol_chamber[self.index_gas]) - self.n_moles_gas_chamber == 0  # type: ignore[reportArgumentType]
        )

        # conservation of total enthalpy in reacting system
        self.opti.subject_to(
            (
                np.sum(self.mol_chamber * species_standard_enthalpies)  # type: ignore[reportArgumentType]
                - self.total_enthalpy_ing
            )
            == 0
        )

        # isentropic, frozen nozzle flow
        self.entropy_chamber = self._molar_entropy(
            temperature=self.temp_chamber,
            pressure=self.p_c,
        )
        self.entropy_exit = self._molar_entropy(
            temperature=self.temp_exit, pressure=self.p_e
        )
        self.opti.subject_to(self.entropy_chamber == self.entropy_exit)

    def _process_prop_formula(self):
        """Process propellant formula for use in optimization."""

        # define some relavant attributes
        self.ingredients = list(self.prop_formula.keys())
        self.n_ingredients = np.size(self.ingredients)
        self.mass_fracs_ing = np.array(list(self.prop_formula.values()))

        # read in prop ingredents yaml
        prop_ingredients_path = self.parent_directory / "helpers/prop_ingredients.yaml"
        with open(prop_ingredients_path) as file:
            prop_ingredients_dict = yaml.safe_load(file)
        self.prop_ingredients_dict = prop_ingredients_dict

        self.prop_elements = thermo.get_elements_in_propellant(
            prop_formula=self.prop_formula,
            prop_ingredients_dict=prop_ingredients_dict,
        )

        # get molecular weights and heats of formation for ingredients
        mol_weight_ing = []
        heat_form_ing = []
        for ingredient in self.ingredients:
            ingredient_dict = prop_ingredients_dict["species"][ingredient]
            mol_weight_ing.append(ingredient_dict["mol_weight"] * 1e-3)
            heat_form_ing.append(ingredient_dict["heat_form"])
        self.mol_weight_ing = np.array(mol_weight_ing)
        heat_form_ing = np.array(heat_form_ing)

        # get total enthalpy for system assuming total system mass is
        # arbitrarily 1 kg, [units: J]; OK - checked with RPA
        self.total_enthalpy_ing = np.sum(  # type: ignore[reportArgumentType]
            heat_form_ing * self.mass_fracs_ing / mol_weight_ing  # type: ignore[reportOperatorIssue]
        )

    def _process_gas_species(self):
        """Process gas species for use in optimization."""

        n_elements = self.gas_obj.n_elements

        # select possible species in combustion products
        selected_species = thermo.get_species_to_include(
            elements_in_prop=self.prop_elements,
            gas_obj=self.gas_obj,
        )
        self.selected_species = selected_species  # TODO consolidate
        selected_species_indices = [
            self.gas_obj.species_index(species) for species in selected_species
        ]
        selected_elements_indices = [
            self.gas_obj.element_index(element) for element in self.prop_elements
        ]

        # call thermo func dict, and re-index to selected species
        thermo_func_dict = thermo.make_thermo_funcs(gas_obj=self.gas_obj)
        self.entropy_funcs = [
            thermo_func_dict["entropy_funcs"][i] for i in selected_species_indices
        ]
        self.enthalpy_funcs = [
            thermo_func_dict["enthalpy_funcs"][i] for i in selected_species_indices
        ]
        self.cp_funcs = [
            thermo_func_dict["heat_cap_funcs"][i] for i in selected_species_indices
        ]

        # get indices of condensed and gaseous species
        condensed_phases = ["AL2O3", "C(gr)", "FeCL3(L)"]
        self.index_condensed = []
        self.index_gas = []
        for index, species in enumerate(selected_species):
            if species in condensed_phases:
                self.index_condensed.append(index)
            else:
                self.index_gas.append(index)

        # make stoich coefficient matrix for reactants: n_species x n_elements
        reac_stoich_coef_mat = np.zeros((self.n_ingredients, n_elements))  # OK
        for index, ingredient in enumerate(self.ingredients):
            species_comp = self.prop_ingredients_dict["species"][ingredient]["formula"]
            for element in species_comp:
                # match index
                element_index = self.gas_obj.element_index(element)
                element_val = species_comp[element]
                reac_stoich_coef_mat[index, element_index] = element_val

        # re-index to selected elements
        reac_stoich_coef_mat = reac_stoich_coef_mat[:, selected_elements_indices]
        self.reac_stoich_coef_mat = reac_stoich_coef_mat
        self.mol_weights_chamber = (  # [kg mol**-1]
            self.gas_obj.molecular_weights[selected_species_indices] * 1e-3
        )

        # make stoich coefficient matrix for products, and re-index
        prod_stoich_coef_mat = thermo.make_stoich_coef_matrix(self.gas_obj)
        prod_stoich_coef_mat = prod_stoich_coef_mat[selected_species_indices, :]
        prod_stoich_coef_mat = prod_stoich_coef_mat[:, selected_elements_indices]
        self.prod_stoich_coef_mat = prod_stoich_coef_mat

    def _process_results(self):
        """Process optimization results into more useful numbers."""

        # get mass and mole fractions
        self.n_tot = np.sum(self.mol_chamber)  # type: ignore[reportOperatorIssue]
        self.mass_fracs = self.mol_chamber * self.mol_weights_chamber
        self.mol_fracs = self.mol_chamber / self.n_tot

        # calculate enthalpy after reaction (should equal enthalpy before
        # reaction)
        self.enthalpy_after_reac = np.sum(  # type: ignore[reportOperatorIssue]
            np.array([func(self.temp_chamber) for func in self.enthalpy_funcs])
            * self.mol_chamber
        )

        # calculate generic gas properties
        self.cp_after_reac = np.sum(  # type: ignore[reportOperatorIssue]
            np.array([func(self.temp_chamber) for func in self.cp_funcs])
            * self.mol_chamber
        )
        self.mean_MW = 1 / self.n_tot
        self.R_gas = R_UNIV / self.mean_MW
        self.gamma = 1 / (1 - self.R_gas / self.cp_after_reac)

    def get_results(self):
        """Return dictionary of important results."""

        results = {
            "species": self.selected_species,
            "mass_frac": self.mass_fracs,
            "mol_frac": self.mol_fracs,
            "T_c": self.temp_chamber,
            "c_p": self.cp_after_reac,
            "R_gas": self.R_gas,
            "gamma": self.gamma,
            "mean_MW": self.mean_MW,
            "gas_obj_products": self.gas_obj,
        }

        return results

    def print_report(self):
        """Method for printing internal ballistics analysis report."""

        mol_frac_dict = {}
        mass_frac_dict = {}
        for index, name in enumerate(self.selected_species):
            mol_frac_dict[name] = np.around(self.mol_fracs[index], 5)
            mass_frac_dict[name] = np.around(self.mass_fracs[index], 5)

        print("Chamber Temperature: ", np.around(self.temp_chamber, 3), " K")
        print(
            "Mean Molecular Weight (inc. condensed): ",
            np.around(self.mean_MW, 6),
            " kg/mol",
        )
        print("Mean Heat Capacity: ", np.around(self.cp_after_reac, 2), " J/kg-K")
        print("Gas Constant: ", np.around(self.R_gas, 3), " J/kg-K")
        print("Gamma: ", np.around(self.gamma, 4))
        print("Mole Fractions:\n", json.dumps(mol_frac_dict, sort_keys=True, indent=4))
        print("Mass Fractions:\n", json.dumps(mass_frac_dict, sort_keys=True, indent=4))
        print("Entropy After Reaction: ", self.entropy_chamber, " J/kg-K")


if __name__ == "__main__":
    ### Simple Example
    oxamide_frac = 0.2
    prop_formula_ox = {
        "AP": 0.8 * (1 - oxamide_frac),
        "HTPB+Curative": 0.142 * (1 - oxamide_frac),
        "IDP": 0.0524 * (1 - oxamide_frac),
        "C(gr)": 0.0026 * (1 - oxamide_frac),
        "HX-752": 0.003 * (1 - oxamide_frac),
        "Oxamide": oxamide_frac,
    }
    p_c = 1e6  # Pa

    chamber1 = ChamberEquilibrium(propellant_formula=prop_formula_ox, p_c=p_c)
    chamber1.print_report()

    ### Example with Sweep

    # establish ranges for sweeping pressure and oxamide mass fraction
    ox_range = np.linspace(0, 0.2, num=20)
    pressure_range = np.array([0.5, 1, 2, 3]) * 1e6  # type: ignore[reportOperatorIssue]

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(9, 6)
    ax.set_xlabel("Propellant Oxamide Mass Fraction [-]")
    ax.set_ylabel("Equilibrium Chamber Temperature [K]")
    fig.suptitle(
        "Chamber Temperature vs. Propellant Oxamide Mass Fraction\nfor Different Chamber Pressures"
    )

    # sweep parameters, run simulation, and plot for each chamber pressure
    for p in pressure_range:
        p_output = []
        for (
            oxamide_frac
        ) in ox_range:  # initialize propellant formula parameter to sweep
            opti = asb.Opti()  # type: ignore[reportCallIssue]
            prop_formula_ox = {
                "AP": 0.8 * (1 - oxamide_frac),
                "HTPB+Curative": 0.142 * (1 - oxamide_frac),
                "IDP": 0.0524 * (1 - oxamide_frac),
                "C(gr)": 0.0026 * (1 - oxamide_frac),
                "HX-752": 0.003 * (1 - oxamide_frac),
                "Oxamide": oxamide_frac,
            }
            chamber2 = ChamberEquilibrium(
                propellant_formula=prop_formula_ox,
                p_c=p,
                opti=opti,  # type: ignore[reportCallIssue]
            )
            sol = opti.solve(verbose=False)  # set verbose to false
            chamber2.substitute_solution(sol)
            p_output.append(chamber2.temp_chamber)

        ax.plot(ox_range, p_output, label=f"{p * 1e-6} MPa")

    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    plt.show()
