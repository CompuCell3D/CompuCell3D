#ifndef ENERGYFUNCTIONCALCULATORTESTDATAGENERATION_H
#define ENERGYFUNCTIONCALCULATORTESTDATAGENERATION_H


#include "EnergyFunctionCalculator.h"
#include <list>
#include <string>
#include <fstream>

namespace CompuCell3D {

    class PottsTestData;

    class Simulator;

    class EnergyFunctionCalculatorTestDataGeneration : public EnergyFunctionCalculator {

    public:
        EnergyFunctionCalculatorTestDataGeneration();

        virtual ~EnergyFunctionCalculatorTestDataGeneration();

        void init(CC3DXMLElement *_xmlData) override {}

        double
        changeEnergy(Point3D &pt, const CellG *newCell, const CellG *oldCell, const unsigned int _flipAttempt) override;

        virtual void get_current_mcs_accepted_mask_npy_array(int *intvec, int n) {}

        virtual void get_current_mcs_prob_npy_array(double *doublevec, int n) {}

        virtual void get_current_mcs_flip_attempt_points_npy_array(short *shortvec, int n) {}

        virtual void setLastFlipAccepted(bool _accept) { lastFlipAccepted = _accept; }

        virtual void set_acceptance_probability(double _prob) {}

        // Python reporting
        std::vector <std::string> getEnergyFunctionNames() { return energyFunctionsNameVec; }

        virtual std::vector <std::vector<double>> getCurrentEnergyChanges() {
            return std::vector < std::vector < double > > ();
        }

        virtual std::vector<bool> getCurrentFlipResults() { return std::vector<bool>(); }

        virtual void log_output(PottsTestData &potts_test_data);


        std::string get_output_file_name();

        std::string get_input_file_name();

    protected:
        std::map<std::string, EnergyFunction *> nameToEnergyFunctionMap;
    private:
        bool header_written = false;


    };

};


#endif

