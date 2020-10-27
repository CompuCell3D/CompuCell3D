from cc3d.core.sim_service import service_roadrunner

"""
This demo teaches usage of the sim_service implementation of RoadRunner, including
    - Ordinary RoadRunner usage through the sim_service implementation
    - How to interactively run a sim_service RoadRunner simulation in Python
"""
__author__ = "T.J. Sego, Ph.D."
__email__ = "tjsego@iu.edu"


model_string = """
model test
    compartment C1;
    C1 = 1.0;
    species S1, S2;

    S1 = 10.0;
    S2 = 0.0;
    S1 in C1; S2 in C1;
    J1: S1 -> S2; k1*S1;

    k1 = 1.0;
end
"""


def main_basic():
    sim_service = service_roadrunner()
    sim_service.run()
    sim_service.load(model_string)
    results = sim_service.simulate(0, 50, 10)
    print(results)
    try:
        print("Trying convenience accessor for S1")
        S1 = sim_service.S1  # This doesn't work for sim_service RoadRunner...
        print(f"S1 = {S1}")
    except AttributeError:
        print("Falling back on reliable accessor for S1")
        S1 = sim_service.getValue('S1')  # ... but this does!
        print(f"S1 = {S1}")


def main_interactive():
    sim_service = service_roadrunner(step_size=5)
    sim_service.run()
    sim_service.load(model_string)
    sim_service.start()
    for x in range(10):
        if x == 5:
            # sim_service.S1 = 1.0  # This doesn't work for sim_service RoadRunner...
            sim_service.setValue('S1', 1.0)  # ... but this does!

        sim_service.step()
        results_step = sim_service.getSimulationData()
        print(results_step)


if __name__ == '__main__':
    main_basic()
    main_interactive()
