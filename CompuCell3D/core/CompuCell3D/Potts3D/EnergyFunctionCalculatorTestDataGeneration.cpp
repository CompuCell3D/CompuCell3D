#include "EnergyFunctionCalculatorTestDataGeneration.h"
#include "EnergyFunction.h"
#include <CompuCell3D/CC3DExceptions.h>
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/PottsParseData.h>
#include "PottsTestData.h"


using namespace CompuCell3D;
using namespace std;

/**
 * Default constructor
 */
EnergyFunctionCalculatorTestDataGeneration::EnergyFunctionCalculatorTestDataGeneration() : EnergyFunctionCalculator() {

}

/**
 * Default destructor
 */
EnergyFunctionCalculatorTestDataGeneration::~EnergyFunctionCalculatorTestDataGeneration() = default;

/**
 * wrapper function that computes change of energy due to pixel copy attempt
 * @param pt change pixel
 * @param newCell cell overwriting new pixel
 * @param oldCell current cell
 * @param _flipAttempt current pixel copy attempt (withing mcs)
 * @return
 */
double EnergyFunctionCalculatorTestDataGeneration::changeEnergy(Point3D &pt, const CellG *newCell, const CellG *oldCell,
                                                                const unsigned int _flipAttempt) {
    double change = 0;
    for (unsigned int i = 0; i < energyFunctions.size(); i++) {

        auto energy = energyFunctions[i]->changeEnergy(pt, newCell, oldCell);
        change += energy;
        energyFunctionNameToValueMap[energyFunctionsNameVec[i]] = energy;
    }
    return change;
}

/**
 * generates full path for test data output file. Called during test data generation
 * @return full path for test data output file
 */
std::string EnergyFunctionCalculatorTestDataGeneration::get_output_file_name() {

    return sim->output_directory + "/" + "potts_data_output.csv";
}

/**
 * generates full path for test data input file. Called during test run
 * @return full path for test data input file
 */
std::string EnergyFunctionCalculatorTestDataGeneration::get_input_file_name() {
    return potts->get_simulation_input_dir() + "/" + "potts_data_output.csv";
}

/**
 * serializes PottsTestData object to a file (path given by get_output_file_name())
 * @param potts_test_data PottsTestData object
 */
void EnergyFunctionCalculatorTestDataGeneration::log_output(PottsTestData &potts_test_data) {

    potts_test_data.energyFunctionNameToValueMap = energyFunctionNameToValueMap;
    std::string file_name = get_output_file_name();

    if (!header_written) {
        header_written = potts_test_data.write_header(file_name);
        if (!header_written) throw CC3DException(" Could not write header to " + file_name);
    }

    bool write_ok = potts_test_data.serialize(file_name);
    if (!write_ok) throw CC3DException(" Could not write data to " + file_name);

}

