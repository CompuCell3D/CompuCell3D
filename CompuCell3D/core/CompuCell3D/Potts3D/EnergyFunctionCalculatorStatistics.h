#ifndef ENERGYFUNCTIONCALCLUATORSTATISTICS_H
#define ENERGYFUNCTIONCALCLUATORSTATISTICS_H

#include "EnergyFunctionCalculator.h"
#include <list>
#include <string>
#include <fstream>

namespace CompuCell3D {

    class EnergyFunctionCalculatorStatistics : public EnergyFunctionCalculator {

    public:
        EnergyFunctionCalculatorStatistics();

        virtual void init(CC3DXMLElement *_xmlData);

        virtual ~EnergyFunctionCalculatorStatistics();

        virtual double
        changeEnergy(Point3D &pt, const CellG *newCell, const CellG *oldCell, const unsigned int _flipAttempt);

        virtual void setLastFlipAccepted(bool _accept);

        virtual void set_acceptance_probability(double _prob);

        virtual void get_current_mcs_accepted_mask_npy_array(int *intvec, int n);

        virtual void get_current_mcs_prob_npy_array(double *doublevec, int n);

        virtual void get_current_mcs_flip_attempt_points_npy_array(short *shortvec, int n);


        virtual long get_number_energy_fcn_calculations();

        virtual void range(int *rangevec, int n);

        // Python reporting

        std::vector <std::vector<double>> getCurrentEnergyChanges() { return totEnergyDataListCurrent; }

        std::vector<bool> getCurrentFlipResults() { return accNotAccListCurrent; }

    private:

        int NTot;
        int NAcc;
        int NRej;
        int lastFlipAttempt;
        long current_mcs_pos; // holds index of the totEnergyDataList's first item for the current mcs
        std::vector<double> lastEnergyVec;

        std::list <Point3D> pixel_copy_attempt_points_list;
        std::list<double> acceptance_probability_list;
        std::list<int> mcs_list;

        // stores energies for each spin flip attempt
        std::list <std::vector<double>> totEnergyDataList;
        //stores energies for each spin flip attempt for current step; for reporting in Python
        std::vector <std::vector<double>> totEnergyDataListCurrent;
        //tells whether entry in totEnergyVecVec is accepted or not
        std::list<bool> accNotAccList;
        // tells whether entry in totEnergyDataListCurrent is accepted or not
        std::vector<bool> accNotAccListCurrent;

        //stat data vectors
        std::vector<double> avgEnergyVectorTot;
        std::vector<double> stdDevEnergyVectorTot;
        std::vector<double> avgEnergyVectorAcc;
        std::vector<double> stdDevEnergyVectorAcc;
        std::vector<double> avgEnergyVectorRej;
        std::vector<double> stdDevEnergyVectorRej;
        int fieldWidth;
        std::string outFileName;
        std::string outFileCoreNameSpinFlips;
        std::ofstream *out;
        std::ofstream *outAccSpinFlip;
        std::ofstream *outRejSpinFlip;
        std::ofstream *outTotSpinFlip;
        int mcs;
        int analysisFrequency;
        int singleSpinFrequency;

        bool wroteHeader;
        bool outputEverySpinFlip;
        bool gatherResultsSpinFlip;
        bool outputAcceptedSpinFlip;
        bool outputRejectedSpinFlip;
        bool outputTotalSpinFlip;


        bool gatherResultsFilesPrepared;

        void writeHeader();

        void writeHeaderFlex(std::ofstream &_out);

        void writeDataLineFlex(std::ofstream &_out, std::vector<double> &_energies);

        void initialize();

        void prepareNextStep();

        void calculateStatData();

        void outputResults();

        void outputResultsSingleSpinFlip();

        void prepareGatherResultsFiles();

        void outputResultsSingleSpinFlipGatherResults();


    };

};


#endif

