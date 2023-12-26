import pybamm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

API_KEY = ""

PARAMETER_VALUES = pybamm.ParameterValues(
    {
        "Thermodynamic Factor": 1.0,
        "Ambient temperature [K]": 298.15,
        "Cation transference number": 0.2594,
        "Cell cooling surface area [m2]": 0.00531,
        "Cell volume [m3]": 2.42e-05,
        "Current function [A]": 5.0,
        "Electrode height [m]": 0.065,
        "Electrode width [m]": 1.58,
        "Initial concentration in electrolyte [mol.m-3]": 1000.0,
        "Initial concentration in negative electrode [mol.m-3]": 29866.0,
        "Initial concentration in positive electrode [mol.m-3]": 17038.0,
        "Initial temperature [K]": 298.15,
        "Lower voltage cut-off [V]": 2.5,
        "Maximum concentration in negative electrode [mol.m-3]": 33133.0,
        "Maximum concentration in positive electrode [mol.m-3]": 63104.0,
        "Negative current collector conductivity [S.m-1]": 58411000.0,
        "Negative current collector density [kg.m-3]": 8960.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 620.7659832284165,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Negative current collector thickness [m]": 1.2e-05,
        "Negative electrode Bruggeman coefficient (electrode)": 1.5,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        "Negative electrode active material volume fraction": 0.75,
        "Negative electrode conductivity [S.m-1]": 215.0,
        "Negative electrode density [kg.m-3]": 1657.0,
        "Negative electrode diffusivity [m2.s-1]": 3.3e-14,
        "Negative electrode electrons in reaction": 1.0,
        "Negative electrode porosity": 0.25,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 1128.6654240516666,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode thickness [m]": 8.52e-05,
        "Negative particle radius [m]": 5.86e-06,
        "Nominal cell capacity [A.h]": 5.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Positive current collector conductivity [S.m-1]": 36914000.0,
        "Positive current collector density [kg.m-3]": 2700.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 1446.3041219633499,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Positive current collector thickness [m]": 1.6e-05,
        "Positive electrode Bruggeman coefficient (electrode)": 1.5,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,
        "Positive electrode active material volume fraction": 0.665,
        "Positive electrode conductivity [S.m-1]": 0.18,
        "Positive electrode density [kg.m-3]": 3262.0,
        "Positive electrode diffusivity [m2.s-1]": 4e-15,
        "Positive electrode electrons in reaction": 1.0,
        "Positive electrode porosity": 0.335,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 1128.6654240516666,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode thickness [m]": 7.56e-05,
        "Positive particle radius [m]": 5.22e-06,
        "Reference temperature [K]": 298.15,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 397.0,
        "Separator porosity": 0.47,
        "Separator specific heat capacity [J.kg-1.K-1]": 1128.6654240516666,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,
        "Separator thickness [m]": 1.2e-05,
        "Total heat transfer coefficient [W.m-2.K-1]": 20.0,
        "Typical current [A]": 5.0,
        "Typical electrolyte concentration [mol.m-3]": 1000.0,
        "Upper voltage cut-off [V]": 4.2,
    },
)

class SimulationRunner(object):
    def __init__(self):
        self.chemistry = 'Chen2020'
        self.model = pybamm.lithium_ion.DFN()
        self.experiment = pybamm.Experiment(["Discharge at 1C until 3.0 V"])
        self.time_steps = 100
        self.max_simulation_time = 3600 # 1 hour max
        self.api_client = API_KEY
        self.output_directory = "datasets"
        self.parameter_values = PARAMETER_VALUES

    def run_dfn_simulation(self):
        self.model = pybamm.BaseModel()

        c = pybamm.Variable("c", domain="unit line")
        k = 3.3e-14

        D = k * (1 + c)
        dcdt = pybamm.div(D * pybamm.grad(c))
        self.model.rhs = {c: dcdt}

        D_right = pybamm.BoundaryValue(D, "right")
        self.model.boundary_conditions = {
            c: {
                "left": (1, "Dirichlet"),
                "right": (1 / D_right, "Neumann")
            }
        }
        
        x = pybamm.SpatialVariable("x", domain="unit line")
        initial_condition_expr = x + 1
        self.model.initial_conditions = {c: initial_condition_expr}

        geometry = self.model.default_geometry

        param = self.model.default_parameter_values
        param.process_model(self.model)
        param.process_geometry(geometry)

        mesh = pybamm.Mesh(geometry, self.model.default_submesh_types, self.model.default_var_pts)

        disc = pybamm.Discretisation(mesh, self.model.default_spatial_methods)
        disc.process_model(self.model)

        t_eval = np.linspace(0, 0.2, 100)
        solution = self.model.default_solver.solve(self.model, t_eval)

        return solution

# Create an instance of SimulationRunner
runner = SimulationRunner()

# Run DFN simulation
solution = runner.run_dfn_simulation()

time = solution["Time [s]"].entries
terminal_voltage = solution["Terminal voltage [V]"].entries  # Replace with your desired variable

# Plotting the variable against time
plt.figure(figsize=(8, 6))
plt.plot(time, terminal_voltage, label='Terminal voltage [V]')  # Replace label with your variable name
plt.xlabel('Time [s]')
plt.ylabel('Terminal voltage [V]')  # Replace with your variable label
plt.title('Terminal Voltage over Time')  # Replace with your title
plt.legend()
plt.grid(True)
plt.show()



