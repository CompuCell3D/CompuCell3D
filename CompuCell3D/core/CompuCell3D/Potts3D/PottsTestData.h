#ifndef POTTSTESTDATA_H
#define POTTSTESTDATA_H

#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/CC3DExceptions.h>
#include <string>
#include <map>
#include <vector>


namespace CompuCell3D {

	class CC3DException;
    class PottsTestDataHeaderSpecs {
    public:
        std::vector <std::string> columns;

        // position at which energy functions will start
        int energy_function_position = 9;

    };

    class PottsTestData {

    public:
        PottsTestData() {}

        Point3D changePixel;
        Point3D changePixelNeighbor;
        double motility;
        bool pixelCopyAccepted;
        double acceptanceFunctionProbability;

        std::map<std::string, double> energyFunctionNameToValueMap;
        bool using_connectivity = false;
        double connectivity_energy = 0.0;

        bool write_header(std::string file_name);

        bool serialize(std::string file_name);

        std::vector <PottsTestData> deserialize_potts_data_sequence(std::ifstream &infile);

        PottsTestData
        deserialize_single_potts_data(std::string line, PottsTestDataHeaderSpecs &potts_test_data_header_specs);

        PottsTestDataHeaderSpecs deserialize_header(std::ifstream &infile);

        std::vector <std::string> split_string(std::string str_to_plit, char delimiter);

        bool compare_potts_data(PottsTestData &potts_data_to_compare) ;

        double relative_difference(double x, double y);

        double abs_difference(double x, double y);

    };

};


#endif

