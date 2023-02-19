#include "PottsTestData.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cmath>

#include <CompuCell3D/CC3DExceptions.h>
#include <PublicUtilities/StringUtils.h>
#include <Logger/CC3DLogger.h>


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
        for (const auto &kv: energyFunctionNameToValueMap) {
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
        out << std::setprecision(6) << std::fixed << motility << ",";
        out << pixelCopyAccepted << ",";
        out << std::setprecision(6) << std::fixed << acceptanceFunctionProbability;

        for (const auto &kv: energyFunctionNameToValueMap) {
            out << "," << std::setprecision(6) << std::fixed << kv.second;
        }
        if (using_connectivity) {
            out << "," << connectivity_energy;
        }

        out << endl;

        return true;
    }
    return false;

}


std::vector <std::string> PottsTestData::split_string(std::string str_to_plit, char delimiter) {

    std::vector <std::string> result;

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

PottsTestDataHeaderSpecs PottsTestData::deserialize_header(std::ifstream &infile) {
    std::string line;
    std::getline(infile, line);

    PottsTestDataHeaderSpecs potts_test_data_headers_specs;
    potts_test_data_headers_specs.columns = split_string(line, ',');

    return potts_test_data_headers_specs;

}

PottsTestData
PottsTestData::deserialize_single_potts_data(std::string line, PottsTestDataHeaderSpecs &potts_test_data_header_specs) {

    std::vector <std::string> line_values = split_string(line, ',');
    PottsTestData potts_test_data;

    potts_test_data.changePixel.x = strToUInt(line_values[0]);
    potts_test_data.changePixel.y = strToUInt(line_values[1]);
    potts_test_data.changePixel.z = strToUInt(line_values[2]);

    potts_test_data.changePixelNeighbor.x = strToUInt(line_values[3]);
    potts_test_data.changePixelNeighbor.y = strToUInt(line_values[4]);
    potts_test_data.changePixelNeighbor.z = strToUInt(line_values[5]);

    potts_test_data.motility = strToDouble(line_values[6]);

    potts_test_data.pixelCopyAccepted = (bool) strToInt(line_values[7]);

    potts_test_data.acceptanceFunctionProbability = strToDouble(line_values[8]);

    size_t possible_connectivity_column_idx = potts_test_data_header_specs.columns.size() - 1;
    size_t max_col_idx = potts_test_data_header_specs.columns.size();
    if (potts_test_data_header_specs.columns[possible_connectivity_column_idx] == "Connectivity") {
        --max_col_idx;
        potts_test_data.using_connectivity = true;
        potts_test_data.connectivity_energy = strToDouble(line_values[possible_connectivity_column_idx]);
    }

    for (unsigned int i = potts_test_data_header_specs.energy_function_position; i < max_col_idx; ++i) {
        potts_test_data.energyFunctionNameToValueMap[potts_test_data_header_specs.columns[i]] = strToDouble(
                line_values[i]);
    }


    return potts_test_data;

}

std::vector <PottsTestData> PottsTestData::deserialize_potts_data_sequence(std::ifstream &infile) {
    //ifstream infile(file_name);
    std::vector <PottsTestData> potts_test_data_vector;
    if (infile) {

        PottsTestDataHeaderSpecs potts_test_data_header_specs = deserialize_header(infile);
        std::string line;
//        int i = 0;
        while (std::getline(infile, line)) {
            PottsTestData potts_test_data = deserialize_single_potts_data(line, potts_test_data_header_specs);
            potts_test_data_vector.push_back(potts_test_data);

//            cerr<<"potts_test_data.energyFunctionNameToValueMap.size()="<<potts_test_data.energyFunctionNameToValueMap.size()<<endl;
//
//            cerr<<"potts_test_data.energyFunctionNameToValueMap[potts_test_data_header_specs.columns[9]]="<<potts_test_data.energyFunctionNameToValueMap[potts_test_data_header_specs.columns[9]]<<endl;
//            cerr<<"potts_test_data.energyFunctionNameToValueMap[potts_test_data_header_specs.columns[10]]="<<potts_test_data.energyFunctionNameToValueMap[potts_test_data_header_specs.columns[10]]<<endl;
//            for (const auto &kv: potts_test_data.energyFunctionNameToValueMap){
//                cerr<<"deserialize"<<endl;
//                cerr<<"kv.first="<<kv.first<<" kv.second="<<kv.second<<endl;
//            }
//            i++;
//            if (i >10){
//                break;
//            }
        }

    }
    return potts_test_data_vector;
}

double PottsTestData::relative_difference(double x, double y) {

    return fabs(x - y) / (x + y + numeric_limits<double>::epsilon());

}

double PottsTestData::abs_difference(double x, double y) {

    return fabs(x - y);

}


bool PottsTestData::compare_potts_data(PottsTestData &potts_data_to_compare) throw(CompuCell3D::CC3DException){

    double tol = 1e-6;
    if (changePixel != potts_data_to_compare.changePixel) throw CC3DException("change pixel is different ");
    if (changePixelNeighbor != potts_data_to_compare.changePixelNeighbor)
        throw CC3DException("change pixel neighbor is different ");
    if (using_connectivity != potts_data_to_compare.using_connectivity)
        throw CC3DException("using_connectivity is different ");
    if (connectivity_energy != potts_data_to_compare.connectivity_energy)
        throw CC3DException("connectivity_energy is different ");

    for (const auto &kv: energyFunctionNameToValueMap) {
        const auto &mitr_computed = potts_data_to_compare.energyFunctionNameToValueMap.find(kv.first);
        if (mitr_computed != potts_data_to_compare.energyFunctionNameToValueMap.end()) {

            double difference_value = abs_difference(kv.second, mitr_computed->second);

            if (difference_value > tol) {
                CC3D_Log(LOG_DEBUG) <<  "detected a difference in " << kv.first << " recorded=" << kv.second << " computed="
                                    << mitr_computed->second;
                CC3D_Log(LOG_DEBUG) << "difference_value=" << difference_value;

                throw CC3DException(string(kv.first) + " energy term different ");
            }
        } else {
            throw CC3DException(string(kv.first) + " energy was not found in the computed energy container");
        }
    }


    if (abs_difference(motility, potts_data_to_compare.motility) >= tol) throw CC3DException("motility is different");
    if (abs_difference(acceptanceFunctionProbability, potts_data_to_compare.acceptanceFunctionProbability) >= 1e-4)
        throw CC3DException("acceptanceFunctionProbability is different");

    return true;
}


//bool PottsTestData::compare_potts_data(PottsTestData &potts_data_to_compare) throw(CompuCell3D::CC3DException){
//
//    double tol = 1e-6;
//    if (changePixel != potts_data_to_compare.changePixel) throw CC3DException("change pixel is different ");
//    if (changePixelNeighbor != potts_data_to_compare.changePixelNeighbor)
//        throw CC3DException("change pixel neighbor is different ");
//    if (using_connectivity != potts_data_to_compare.using_connectivity)
//        throw CC3DException("using_connectivity is different ");
//    if (connectivity_energy != potts_data_to_compare.connectivity_energy)
//        throw CC3DException("connectivity_energy is different ");
////    cerr<<"FRESH COMPARE"<<endl;
////    cerr<<" potts_data_to_compare.energyFunctionNameToValueMap.size()"<< potts_data_to_compare.energyFunctionNameToValueMap.size()<<endl;
////    cerr<<"energyFunctionNameToValueMap.size()="<<energyFunctionNameToValueMap.size()<<endl;
//
//
//////    for (const auto &kv: energyFunctionNameToValueMap)
////    map<int, int> my_map;
////    my_map[0] = 0;
////    my_map[1] = 1;
////    cerr<<"my_map.size()="<<my_map.size()<<endl;
////    int j = 0;
////    for (const auto &kv: my_map){
////        cerr<<"j="<<j<<endl;
////        ++j;
////
////        cerr<<"f="<<kv.first<<" v="<<kv.second<<endl;
////    }
//    int i = 0;
//    for (auto & mitr : energyFunctionNameToValueMap){
////        cerr<<"i="<<i<<endl;
////        ++i;
//        const auto &mitr_computed = potts_data_to_compare.energyFunctionNameToValueMap.find(mitr.first);
//        if (mitr_computed != potts_data_to_compare.energyFunctionNameToValueMap.end()){
////            cerr<<"found mitr"<<endl;
//            double difference_value = abs_difference(mitr.second, mitr_computed->second);
////            cerr<<"difference_value="<<difference_value<<endl;
//            if (difference_value > tol) {
//                CC3D_Log(LOG_DEBUG) <<  "detected a difference in " << mitr.first << " recorded=" << mitr.second << " computed="
//                                    << mitr_computed->second;
//                CC3D_Log(LOG_DEBUG) << "difference_value=" << difference_value;
//                throw CC3DException(string(mitr.first) + " energy term different ");
//            }
//        }else{
////            cerr<<"DID NOT FIND MITR"<<endl;
//            throw CC3DException(string(mitr.first) + " energy was not found in the computed energy container");
//        }
//    }
//
//    if (abs_difference(motility, potts_data_to_compare.motility) >= tol) {
//        throw CC3DException("motility is different");
//    }
//
//    if (abs_difference(acceptanceFunctionProbability, acceptanceFunctionProbability) >= 1e-4){
//            throw CC3DException("acceptanceFunctionProbability is different");
//    }
//    return true;
//
//    for (auto & mitr : energyFunctionNameToValueMap)
//    {
//        cerr<<"i="<<i<<endl;
//        ++i;
//        if (i>energyFunctionNameToValueMap.size()) continue;
//        cerr<<"FIRST check kv.first="<<mitr.first<<" kv.second="<<mitr.second<<endl;
//        cerr<<"energyFunctionNameToValueMap.size()="<<energyFunctionNameToValueMap.size()<<endl;
////        const auto &mitr_computed = potts_data_to_compare.energyFunctionNameToValueMap.find(mitr->first);
////        if (mitr_computed != potts_data_to_compare.energyFunctionNameToValueMap.end()){
////            cerr<<"first found"<<endl;
////            double difference_value = abs_difference(mitr->second, mitr_computed->second);
////            cerr<<"difference_value="<<difference_value<<endl;
//////            if (difference_value > tol) {
////////                CC3D_Log(LOG_DEBUG) <<  "detected a difference in " << kv.first << " recorded=" << kv.second << " computed="
////////                                    << mitr_computed->second;
////////                CC3D_Log(LOG_DEBUG) << "difference_value=" << difference_value;
//////
//////                throw CC3DException(string(kv.first) + " energy term different ");
//////            }
////        }else{
////            cerr<<"first NOT found"<<endl;
////        }
//    }
//
////    for (const auto &kv: energyFunctionNameToValueMap) {
////        const auto &mitr_computed = potts_data_to_compare.energyFunctionNameToValueMap.find(kv.first);
//////        for (const auto &kv1: energyFunctionNameToValueMap){
//////            cerr<<"kv1.first="<<kv1.first<<" kv1.second="<<kv1.second<<endl;
//////        }
////
////        cerr<<"kv.first="<<kv.first<<" kv.second="<<kv.second<<endl;
////        if (mitr_computed != potts_data_to_compare.energyFunctionNameToValueMap.end()) {
////
////            double difference_value = abs_difference(kv.second, mitr_computed->second);
////
////            if (difference_value > tol) {
////                CC3D_Log(LOG_DEBUG) <<  "detected a difference in " << kv.first << " recorded=" << kv.second << " computed="
////                     << mitr_computed->second;
////                CC3D_Log(LOG_DEBUG) << "difference_value=" << difference_value;
////
////                throw CC3DException(string(kv.first) + " energy term different ");
////            }
////        } else {
////            throw CC3DException(string(kv.first) + " energy was not found in the computed energy container");
////        }
////    }
//
////
////    if (abs_difference(motility, potts_data_to_compare.motility) >= tol) {
////        throw CC3DException("motility is different");
////    }
////
////    if (abs_difference(acceptanceFunctionProbability, acceptanceFunctionProbability) >= 1e-4){
////        cerr<<"computted acceptanceFunctionProbability="<<acceptanceFunctionProbability<<" potts_data_to_compare.acceptanceFunctionProbability="<<potts_data_to_compare.acceptanceFunctionProbability<<endl;
////        cerr<<"difference="<<abs_difference(acceptanceFunctionProbability, potts_data_to_compare.acceptanceFunctionProbability)<<endl;
////        cerr<<"difference check "<<(abs_difference(acceptanceFunctionProbability, potts_data_to_compare.acceptanceFunctionProbability) >= 1e-4)<<endl;
////        if ((abs_difference(acceptanceFunctionProbability, potts_data_to_compare.acceptanceFunctionProbability) >= 1e-4)){
////            cerr<<"DIFFERENCE IS GREATER that tolerance"<<endl;
////            cerr<<"Another check acceptanceFunctionProbability="<<acceptanceFunctionProbability<<" potts_data_to_compare.acceptanceFunctionProbability="<<potts_data_to_compare.acceptanceFunctionProbability<<endl;
////            cerr<<"another difference check "<<(abs_difference(acceptanceFunctionProbability, potts_data_to_compare.acceptanceFunctionProbability) >= 1e-4)<<endl;
////            throw CC3DException("acceptanceFunctionProbability is different");
////        }
////
////    }
//
//
//}
