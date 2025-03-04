# Chamber Equilibrium Model

An end-to-end differentiable thermodynamic equilibrium model and solver for solid rocket motors, compatible with [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) and other automatic-differentiation frameworks.


## Installation and Running

First, install the [uv package manager for python](https://docs.astral.sh/uv/getting-started/installation/#installation-methods). 

Then,
```bash
git clone https://github.com/kjmath/rocket-snippets.git
cd rocket-snippets/src/chamber_equilibrium_model
uv run chamber_equilibrium.py
```
Running the model with ```uv run``` will automatically download and install compatible versions of python and all dependencies in a virtual environment. 

## File Structure of this Sub-Module

- ```chamber_equilibrium.py```: the main file, which implements variables and constraints of the thermodynamic equilibrium problem and solves for the equilibrium condition in the ```ChamberEquilibrium()``` class
- ```helpers/thermo.py```: various helper methods, including methods for determining stoichiometric coefficient matrices and implementing [NASA-9 models](https://cantera.org/dev/reference/thermo/species-thermo.html#the-nasa-9-coefficient-polynomial-parameterization) for thermodynamic properties
- ```helpers/products.yaml```: file containing [NASA-9 coefficients](https://ntrs.nasa.gov/citations/20020085330) and other parameters for products in propellants, formatted as a ```yaml``` file for use with ```Cantera``` objects
- ```helpers/prop_ingredients.yaml```: file containing information about molecular weight, heat of formation, and chemical formula for different propellant ingredients
- ```../../tests/test_chamber_equilibrium.py```: unit tests for validating the model, implemented using ```pytest```; outputs are validated against sample outputs from the commercial [RPA](https://www.rocket-propulsion.com/index.htm) software 


## Example Use-Case

This use-case is implemented in the ```chamber_equilibrium.py``` file. 

Some context for this example: This code was written as part of my [PhD work](https://hdl.handle.net/1721.1/151348) to optimize the design of solid rocket motors with constraints on plume radiant emission. The work focused on a class of slow-burning, solid rocket propellants which were doped with the burn-rate suppressant oxamide. 

In this example, we use the model to understand how the equilibrium chamber temperature is affected by varying chamber pressure and mass fractions of the burn-rate suppressant oxamide. A base propellant formulation that can be diluted with some mass fraction ```oxamide_frac``` of oxamide is used:
```python
    prop_formula_ox = {
        "AP": 0.8 * (1 - oxamide_frac),
        "HTPB+Curative": 0.142 * (1 - oxamide_frac),
        "IDP": 0.0524 * (1 - oxamide_frac),
        "C(gr)": 0.0026 * (1 - oxamide_frac),
        "HX-752": 0.003 * (1 - oxamide_frac),
        "Oxamide": oxamide_frac,
    }
```
Oxamide mass fractions from 0 to 0.2 are swept, along with several different values of chamber pressure. The results are plotted below. Increasing oxamide mass fraction significantly reduces the equilibrium chamber temperature for a propellant, while the effects of the operating chamber pressure are essentially negligible by comparison. 


![temp-plot](/docs/images/temp_vs_oxamide_frac.png)

## Governing Equations
This model solves for combustion temperature and products by minimizing their Gibb's free energy subject to conservatinn of mass and enthalpy. 
A formulation of the thermodynamic equilibrium problem that uses Lagrange multipliers (following the formulation given by [Ponomarenko](http://www.rocket-propulsion.info/resources/software/rpa/RPA_LiquidRocketEngineAnalysis.pdf)) so that it can be solved as a system of constrained equations, as shown below.
Species thermodynamic properties are modeled using the NASA 9-coefficient polynomial parameterization.

Minimization of Gibbs free energy for gaseous products:

$$ \hat{g^0_j}(T_c) + \hat{R} T_c \ln \left(\frac{n_j}{n_{tot}} \right) + \hat{R} T_c \ln \left( \frac{p_c}{p_0} \right) - \sum_{i=1}^{N_{elements}} \lambda_{i} a_{ij} = 0 $$

$$ \text{for } j = 1, \ldots, N_{prod} $$

Conservation of mass of chemical elements:

$$ \sum_{j=1}^{N_{prod}} a_{ij} n_j - \sum_{k=1}^{N_{reac}} b_{ik} n_k = 0 \hspace{0.3in} \text{for } i = 1, \ldots, N_{elements} $$

Conservation of enthalpy:

$$ \sum_{j=1}^{N_{prod}} n_j \hat{h}_j^0 - H_0 = 0 $$

Enforcement of molar sum of gaseous products:

$$ \sum_{j=1}^{N_{prod}} n_j - n_{tot} = 0 $$

The parameters in the governing equations are given in the table below:

| Variable        | Python Variable | Description |
| -----------     | -----------     | ----------- |
| $$i$$           | ```i``` |chemical elements       |
| $$j$$           | ```j``` | product species        |
| $$k$$           | ```k```| reactant components      |
| $$N_{elements}$$|```n_elements```|number of chemical elements $i$|
| $$N_{prod}$$    |```n_prod``` |number of product species $j$|
| $$N_{reac}$$    |```n_reac``` |number of reactant components $k$|
| $$T_c$$         |```temp_c```|chamber temperature [K] |
| $$p_c$$         |```p_c``` |chamber pressure [Pa]|
| $$p_0$$         |```p_0``` |standard pressure [Pa] |
| $$\hat{g^0_j}$$ |```g_j``` |molar Gibbs free energy of species $j$ [J mol<sup>-1</sup>]|
| $$\hat{h^0_j}$$ |```h_j``` |molar enthalpy of species $j$ [J mol<sup>-1</sup>]|
| $$\hat{s^0_j}$$ |```s_j``` |molar entropy of species $j$ [J mol<sup>-1</sup> K<sup>-1</sup>]|
| $$H_0$$         |```H_0``` |total system enthalpy [J]|
| $$\lambda_{i}$$ |```lagrange_i``` |Lagrange multiplier for element $i$ |
| $$a_{ij}$$ |```prod_stoich_coef_mat```|number of atoms of element $i$ per mole of product species $j$|
| $$b_{ik}$$ |```reac_stoich_coef_mat``` |number of atoms of element $i$ per mole of reactant component $k$|
| $$n_j$$ |```n_j``` |number of moles of product species $j$|
| $$n_k$$ |```n_k``` |number of moles of reactant component $k$|
| $$n_{tot}$$ |```n_tot``` |total number of moles of product species|
| $$\hat{R}$$ |```R_univ```|universal gas constant [J mol<sup>-1</sup> K<sup>-1</sup>]|

Gibbs free energy can be calculated using $\hat{g^0_j} = \hat{h^0_j} - T_c \hat{s^0_j}$.
Lagrange multipliers $\lambda_i$ are introduced, following the procedure used by Ponomarenko.

## Validation

The model outputs were validated against the outputs of the Rocket Propulsion Analysis ([RPA](https://www.rocket-propulsion.com/index.htm)) software.
For each test case, the chamber pressure and propellant formulation were pre-selected, and resulting chamber temperature and species mole fractions are then compared.
The relative errors of the model outputs are ≤ 10<sup>-4</sup>.
The cases below are implemented as unit tests in [```test_chamber_equilibrium.py```](https://github.com/kjmath/rocket-snippets/blob/main/tests/test_chamber_equilibrium.py).


### Case 1

**Inputs:**

Chamber Pressure: 1 MPa

Propellant Formulation: 
| Chemical name                                                       | Mass fraction |
|:--------------------------------------------------------------------|-------------:|
| Ammonium Perchlorate (AP)                                           |         0.8  |
| Hydroxyl Terminated Polybutadiene (HTPB) + Curative                 |       0.142  |
| Isodecyl Pelargonate (IDP)                                          |      0.0524  |
| Graphite powder                                                     |      0.0026  |
| HX-752                                                              |       0.003  |


**Output Comparison:**

| Parameter           | This Model | RPA      | Relative error [-]     |
|---------------------|------------|----------|------------------------|
| Temperature [K]     | 2180.20    | 2180.31  | 5 × 10<sup>-5</sup>    |
| w<sub>CO</sub> [-]  | 0.25063    | 0.25062  | 4 × 10<sup>-5</sup>    |
| w<sub>CO₂</sub> [-] | 0.05549    | 0.05549  | < 1 × 10<sup>-5</sup>    |
| w<sub>H₂</sub> [-]  | 0.21688    | 0.21687  | 5 × 10<sup>-5</sup>    |
| w<sub>H₂O</sub> [-] | 0.25027    | 0.25027  | < 1 × 10<sup>-5</sup>    |
| w<sub>HCl</sub> [-] | 0.14945    | 0.14945  | < 1 × 10<sup>-5</sup>    |
| w<sub>N₂</sub> [-]  | 0.07604    | 0.07603  | 1 × 10<sup>-4</sup>    |

### Case 2

**Inputs:**

Chamber Pressure: 5 MPa

Propellant Formulation: 
| Chemical name                                                       | Mass fraction |
|:--------------------------------------------------------------------|-------------:|
| Ammonium Perchlorate (AP)                                           |         0.8  |
| Hydroxyl Terminated Polybutadiene (HTPB) + Curative                 |       0.142  |
| Isodecyl Pelargonate (IDP)                                          |      0.0524  |
| Graphite powder                                                     |      0.0026  |
| HX-752                                                              |       0.003  |

**Output Comparison:**

| Parameter            | This Model         | RPA       | Relative error [-]    |
|----------------------|---------------------|-----------|-----------------------|
| Temperature [K]      | 2183.96            | 2184.08   | 5 × 10<sup>-5</sup>   |
| w<sub>CO</sub> [-]   | 0.25079            | 0.25079   | < 1 × 10<sup>-5</sup>   |
| w<sub>CO₂</sub> [-]  | 0.05543            | 0.05543   | < 1 × 10<sup>-5</sup>   |
| w<sub>H₂</sub> [-]   | 0.21694            | 0.21691   | 1 × 10<sup>-4</sup>   |
| w<sub>H₂O</sub> [-]  | 0.25050            | 0.25051   | 4 × 10<sup>-5</sup>   |
| w<sub>HCl</sub> [-]  | 0.14970            | 0.14971   | 7 × 10<sup>-5</sup>   |
| w<sub>N₂</sub> [-]   | 0.07606            | 0.07505   | 1 × 10<sup>-4</sup>   |

### Case 3

**Inputs:**

Chamber Pressure: 2 MPa

Propellant Formulation: 
| Chemical name                                                       | Mass fraction |
|:--------------------------------------------------------------------|-------------:|
| Ammonium Perchlorate (AP)                                           |       0.736  |
| Hydroxyl Terminated Polybutadiene (HTPB) + Curative                 |     0.13064  |
| Isodecyl Pelargonate (IDP)                                          |    0.048208  |
| Graphite powder                                                     |    0.002392  |
| HX-752                                                              |     0.00276  |
| Oxamide                                                             |        0.08  |

**Output Comparison:**

| Parameter            | This Model                 | RPA       | Relative error [-]     |
|----------------------|-----------------------------|-----------|------------------------|
| Temperature [K]      | 1904.04                     | 1904.16   | 6 × 10<sup>-5</sup>    |
| w<sub>CO</sub> [-]   | 0.25940                     | 0.25940   | < 1 × 10<sup>-5</sup>  |
| w<sub>CO₂</sub> [-]  | 0.05607                     | 0.05607   | < 1 × 10<sup>-5</sup>    |
| w<sub>H₂</sub> [-]   | 0.24101                     | 0.24099   | 8 × 10<sup>-5</sup>    |
| w<sub>H₂O</sub> [-]  | 0.21998                     | 0.21998   | < 1 × 10<sup>-5</sup>   |
| w<sub>HCl</sub> [-]  | 0.13517                     | 0.13518   | 7 × 10<sup>-5</sup>    |
| w<sub>N₂</sub> [-]   | 0.08823                     | 0.08822   | 1 × 10<sup>-4</sup>    |
