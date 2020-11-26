#include "PottsTestData.h"
#include <fstream> 
#include <sstream>
#include <BasicUtils/BasicString.h>
#include <BasicUtils/BasicException.h>


using namespace CompuCell3D;
using namespace std;


bool PottsTestData::write_header(std::string file_name) {

    ofstream out(file_name, std::ofstream::out);

    if (out) {
        out << "change_pixel_x,";
        out << "change_pixel_y,";
        out << "change_pixel_z,";
        out << "neighbor_change_pixel_x,";
        out << "neighbor_change_pixel_y,";
        out << "neighbor_change_pixel_z,";
        out << "motility,";
        out << "pixel_copy_accepted,";
        out << "acceptance_function_probability";
        for (const auto& kv : energyFuctionNametoValueMap) {
            out << "," << kv.first;
        }
        if (using_connectivity) {
            out << "," << "Connectivity";
        }
        out << endl;
        return true;
    }
    return false;
}

bool PottsTestData::serialize(std::string file_name) {
    ofstream out(file_name, std::ofstream::app);
    if (out) {
        out << changePixel.x << ",";
        out << changePixel.y << ",";
        out << changePixel.z << ",";
        out << changePixelNeighbor.x << ",";
        out << changePixelNeighbor.y << ",";
        out << changePixelNeighbor.z << ",";
        out << motility << ",";
        out << pixelCopyAccepted << ",";        
        out << acceptanceFunctionProbability;

        for (const auto& kv : energyFuctionNametoValueMap) {
            out << "," << kv.second;
        }
        if (using_connectivity) {
            out << "," << connectivity_energy;
        }

        out << endl;

        return true;
    }
    return false;

}
//
//PottsTestData PottsTestData::deserialize(std::ifstream & in){
//    PottsTestData potts_test_data;
//
//    
//}

std::vector<std::string> PottsTestData::split_string(std::string str_to_plit, char delimiter) {

    std::vector<std::string> result;

    //create string stream from the string
    stringstream s_stream(str_to_plit);

    while (s_stream.good()) {
        std::string substr;

        //get first string delimited by comma
        getline(s_stream, substr, delimiter); 
        result.push_back(substr);
    }

    return result;
}

PottsTestDataHeaderSpecs PottsTestData::deserialize_header(std::ifstream & infile) {
    std::string line;
    std::getline(infile, line);

    PottsTestDataHeaderSpecs potts_test_data_headers_specs;
    potts_test_data_headers_specs.columns = split_string(line, ',');

    return potts_test_data_headers_specs;

}

PottsTestData PottsTestData::deserialize_single_potts_data(std::string line, PottsTestDataHeaderSpecs & potts_test_data_header_specs) {

    std::vector<std::string> line_values = split_string(line, ',');
    PottsTestData potts_test_data;

    potts_test_data.changePixel.x = BasicString::parseUInteger(line_values[0]);
    potts_test_data.changePixel.y = BasicString::parseUInteger(line_values[1]);
    potts_test_data.changePixel.z = BasicString::parseUInteger(line_values[2]);

    potts_test_data.changePixelNeighbor.x = BasicString::parseUInteger(line_values[3]);
    potts_test_data.changePixelNeighbor.y = BasicString::parseUInteger(line_values[4]);
    potts_test_data.changePixelNeighbor.z = BasicString::parseUInteger(line_values[5]);

    potts_test_data.motility = BasicString::parseDouble(line_values[6]);

    potts_test_data.pixelCopyAccepted = (bool)BasicString::parseInteger(line_values[7]);
        
    potts_test_data.acceptanceFunctionProbability = BasicString::parseDouble(line_values[8]);
    
    size_t possible_connectivity_column_idx = potts_test_data_header_specs.columns.size() - 1;
    size_t max_col_idx = potts_test_data_header_specs.columns.size();
    if (potts_test_data_header_specs.columns[possible_connectivity_column_idx] == "Connectivity") {
        --max_col_idx;
        potts_test_data.using_connectivity = true;
        potts_test_data.connectivity_energy = BasicString::parseDouble(line_values[possible_connectivity_column_idx]);
    }

    for (unsigned int i = potts_test_data_header_specs.energy_function_position; i < max_col_idx; ++i) {
        potts_test_data.energyFuctionNametoValueMap[potts_test_data_header_specs.columns[i]] = BasicString::parseDouble(line_values[i]);
    }

    

    return potts_test_data;

}

std::vector<PottsTestData> PottsTestData::deserialize_potts_data_sequence(std::ifstream & infile) {
    //ifstream infile(file_name);
    std::vector<PottsTestData> potts_test_data_vector;
    if (infile) {

        PottsTestDataHeaderSpecs potts_test_data_header_specs = deserialize_header(infile);
        std::string line;
        while (std::getline(infile, line)) {
            PottsTestData potts_test_data = deserialize_single_potts_data(line, potts_test_data_header_specs);
            potts_test_data_vector.push_back(potts_test_data);
        }

    }
    return potts_test_data_vector;
}

double PottsTestData::relative_difference(double x, double y) {
    
    return fabs(x - y) / (x + y + std::numeric_limits<double>::epsilon());
    
}

bool PottsTestData::compare_potts_data(PottsTestData & potts_data_to_compare) {
    
    ASSERT_OR_THROW("change pixel is different ", changePixel == potts_data_to_compare.changePixel);
    ASSERT_OR_THROW("change pixel neighbor is different ", changePixelNeighbor == potts_data_to_compare.changePixelNeighbor);    
    ASSERT_OR_THROW("using_connectivity is different ", using_connectivity == potts_data_to_compare.using_connectivity);

    ASSERT_OR_THROW("connectivity_energy is different ", connectivity_energy == potts_data_to_compare.connectivity_energy);
        
    for (const auto& kv : energyFuctionNametoValueMap) {    
        const auto & mitr_computed = potts_data_to_compare.energyFuctionNametoValueMap.find(kv.first);
        if (mitr_computed != potts_data_to_compare.energyFuctionNametoValueMap.end()) {

            double relative_difference_value = relative_difference(kv.second, mitr_computed->second);

            if (relative_difference_value > 1e-4) {
                cerr << "detected a difference in " << kv.first << " recorded=" << kv.second << " computed=" << mitr_computed->second << endl;
                cerr << "relative_difference_value=" << relative_difference_value << endl;

                ASSERT_OR_THROW(string(kv.first) + " energy term different ", false);
            }
        }
        else {
            ASSERT_OR_THROW(string(kv.first) + " energy was not found in the computed energy container", false);
        }
    }
        

    ASSERT_OR_THROW("motility is different", relative_difference(motility , potts_data_to_compare.motility) < 1e-4);    
    ASSERT_OR_THROW("acceptanceFunctionProbability is different", relative_difference(acceptanceFunctionProbability, potts_data_to_compare.acceptanceFunctionProbability)< 1e-4);
    
}
