#ifndef ENERGYFUNCTIONCALCLUATOR_H
#define ENERGYFUNCTIONCALCLUATOR_H

#include <string>
#include <vector>
#include <map>


class CC3DXMLElement;

namespace CompuCell3D {

    class EnergyFunction;

    class CellG;

    class Point3D;

    class Potts3D;

    class Simulator;

    class ParseData;

    class PottsTestData;

    class EnergyFunctionCalculator {

    public:
        EnergyFunctionCalculator();

        virtual ~EnergyFunctionCalculator();

        virtual void init(CC3DXMLElement *_xmlData) {}

        /**
         * Registers energy function plugin with the calculator
         * @param _function obj derived from EnergyFunction class
         */
        virtual void registerEnergyFunction(EnergyFunction *_function);

        /**
         * Registers energy function plugin with the calculator
         * @param _function obj derived from EnergyFunction class
         * @param _functionName energy function plugin name
         */
        virtual void registerEnergyFunctionWithName(EnergyFunction *_function, std::string _functionName);

        /**
        * Unregisters energy function plugin
        * @param _functionName energy function plugin name
        */
        virtual void unregisterEnergyFunction(std::string _functionName);

        virtual void configureEnergyCalculator(std::vector <std::string> &_configVector) {}

        virtual long get_number_energy_fcn_calculations() { return 0; }

        virtual void range(int *rangevec, int n) {}

        /**
         * Computes total energy change due to pixel copy. Sums contributions to delta E from all registered energy
         * functions
         * @param pt change pixel
         * @param newCell cell trying to overwrite change pixel
         * @param oldCell cell currently present at the change pixel
         * @param _flipAttempt number of pixel copy attempt
         * @return delta E
         */
        virtual double changeEnergy(Point3D &pt, const CellG *newCell,
                                    const CellG *oldCell, const unsigned int _flipAttempt);

        virtual void setPotts(Potts3D *_potts) { potts = _potts; }

        virtual void setSimulator(Simulator *_sim) { sim = _sim; }

        /**
         *  returns numpy array proxy with dimension equal to number of pixel copy attempts in current MCS and values
         *  being booleans that denote if pixel copy was accepted or not
         * @param intvec [in, out]
         * @param n
         */
        virtual void get_current_mcs_accepted_mask_npy_array(int *intvec, int n) {}

        virtual void get_current_mcs_prob_npy_array(double *doublevec, int n) {}

        virtual void get_current_mcs_flip_attempt_points_npy_array(short *shortvec, int n) {}

        /**
         * sets las pixel copy accepted flag
         * @param _accept flag - determines if last pixel copy was accepted or not
         */
        virtual void setLastFlipAccepted(bool _accept) { lastFlipAccepted = _accept; }

        virtual void set_acceptance_probability(double _prob) {}

        // Python reporting
        /**
         * Returns Energy Function names
         * @return vector of energy function names
         */
        std::vector <std::string> getEnergyFunctionNames() { return energyFunctionsNameVec; }

        /**
         * returns vector of energy changes for current pixel copy attempt
         * @return
         */
        virtual std::vector <std::vector<double>>
        getCurrentEnergyChanges() { return std::vector < std::vector < double > > (); }

        virtual std::vector<bool> getCurrentFlipResults() { return std::vector<bool>(); }

        /**
         * Returns dictionary energyFunctionName: energyChange
         * @return dictionary energyFunctionName: energyChange
         */
        virtual std::map<std::string, double> getEnergyFunctionNameToValueMap() { return energyFunctionNameToValueMap; }

        /**
         * virtual function used in some derived classes - used to serialize PottsTestData objects
         * @param potts_test_data reference to PottsTestData objects
         */
        virtual void log_output(PottsTestData &potts_test_data) {};


    protected:
        std::vector<EnergyFunction *> energyFunctions;
        std::vector <std::string> energyFunctionsNameVec;

        std::map<std::string, EnergyFunction *> nameToEnergyFunctionMap;
        std::map<std::string, double> energyFunctionNameToValueMap;
        Potts3D *potts;
        Simulator *sim;

        bool lastFlipAccepted;

        /**
         * Validates energy function and current internal state. 
         * Throws a CC3DException if validation fails. 
         * 
         * @param _function obj derived from EnergyFunction class
         */
        void checkEnergyFunction(EnergyFunction *_function);

    };

};


#endif

