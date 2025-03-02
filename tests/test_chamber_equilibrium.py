import pytest
from chamber_equilibrium_model import chamber_equilibrium as ce


# test parameters for ChamberEquilibrium class
# (p_c, prop_formula, T_c_test, mol_fracs_test)
test_params = [
    (
        1e6,
        {
            "AP": 0.8,
            "HTPB+Curative": 0.142,
            "IDP": 0.0524,
            "C(gr)": 0.0026,
            "HX-752": 0.003,
        },
        2180.31,
        {
            "H2O": 0.25027,
            "CO2": 0.05549,
            "CO": 0.25062,
            "HCl": 0.14945,
            "N2": 0.07603,
            "H2": 0.21687,
        },
    ),
    (
        5e6,
        {
            "AP": 0.8,
            "HTPB+Curative": 0.142,
            "IDP": 0.0524,
            "C(gr)": 0.0026,
            "HX-752": 0.003,
        },
        2184.0793,
        {
            "H2O": 0.25051,
            "CO2": 0.05543,
            "CO": 0.25079,
            "HCl": 0.14971,
            "N2": 0.07605,
            "H2": 0.21691,
        },
    ),
    (
        2e6,
        {
            "AP": 0.736,
            "HTPB+Curative": 0.13064,
            "IDP": 0.048208,
            "C(gr)": 0.002392,
            "HX-752": 0.00276,
            "Oxamide": 0.08,
        },
        1904.1612,
        {
            "H2O": 0.21998,
            "CO2": 0.05607,
            "CO": 0.25940,
            "HCl": 0.13518,
            "N2": 0.08822,
            "H2": 0.24099,
        },
    ),
    (
        2e6,
        {
            "AP": 0.736,
            "HTPB+Curative": 0.13064,
            "IDP": 0.048208,
            "C(gr)": 0.002392,
            "HX-752": 0.00276,
            "Al": 0.08,
        },
        2646.106,
        {
            "H2O": 0.16712,
            "CO2": 0.02442,
            "CO": 0.27028,
            "HCl": 0.14157,
            "N2": 0.073184,
            "H2": 0.28079,
            "Al2O3(b)": 0.033831,
        },
    ),
]


@pytest.mark.parametrize("p_c, prop_formula, T_c_test, mol_fracs_test", test_params)
def test_chamber_equilibrium(
    p_c: float,
    prop_formula: dict,
    T_c_test: float,
    mol_fracs_test: dict,
):
    chamber = ce.ChamberEquilibrium(
        propellant_formula=prop_formula,
        p_c=p_c,
    )
    assert chamber.temp_chamber == pytest.approx(T_c_test, rel=0.001)
    for s in mol_fracs_test:
        index = chamber.selected_species.index(s)
        assert chamber.mol_fracs[index] == pytest.approx(mol_fracs_test[s], rel=0.01)
