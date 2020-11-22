#ifndef ENERGYFUNCTIONCALCULATORTESTDATAGENERATION_H
#define ENERGYFUNCTIONCALCULATORTESTDATAGENERATION_H


#include "EnergyFunctionCalculator.h"
#include <list>
#include <string>
#include <fstream>

namespace CompuCell3D{

    class PottsTestData;

class EnergyFunctionCalculatorTestDataGeneration: public EnergyFunctionCalculator{

   public:
       EnergyFunctionCalculatorTestDataGeneration();
       virtual ~EnergyFunctionCalculatorTestDataGeneration();

       virtual void init(CC3DXMLElement *_xmlData) {}
       virtual double changeEnergy(Point3D &pt, const CellG *newCell, const CellG *oldCell, const unsigned int _flipAttempt);
       void setPotts(Potts3D * _potts) { potts = _potts; }
       void setSimulator(Simulator * _sim) { sim = _sim; }

       virtual void  get_current_mcs_accepted_mask_npy_array(int * intvec, int n) {}
       virtual void  get_current_mcs_prob_npy_array(double * doublevec, int n) {}
       virtual void  get_current_mcs_flip_attempt_points_npy_array(short * shortvec, int n) {}

       virtual void setLastFlipAccepted(bool _accept) { lastFlipAccepted = _accept; }
       virtual void set_acecptance_probability(double _prob) { }

       // Python reporting
       std::vector<std::string> getEnergyFunctionNames() { return energyFunctionsNameVec; }
       virtual std::vector<std::vector<double> > getCurrentEnergyChanges() { return std::vector<std::vector<double> >(); }
       virtual std::vector<bool> getCurrentFlipResults() { return std::vector<bool>(); }
       virtual void log_output(PottsTestData &potts_test_data);

protected:
    std::map<std::string, EnergyFunction *> nameToEnergyFuctionMap;    
private:
    bool header_written = false;
    //void write_header();    
    std::string get_output_file_name();

};

};


#endif

