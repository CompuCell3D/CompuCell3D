#ifndef ENERGYFUNCTIONCALCLUATORSTATISTICS_H
#define ENERGYFUNCTIONCALCLUATORSTATISTICS_H

#include "EnergyFunctionCalculator.h"
#include <list>
#include <string>
#include <fstream>

namespace CompuCell3D{

class EnergyFunctionCalculatorStatistics:public EnergyFunctionCalculator{

   public:
      EnergyFunctionCalculatorStatistics();
		virtual void init(CC3DXMLElement *_xmlData);
      virtual ~EnergyFunctionCalculatorStatistics();
      virtual double changeEnergy(Point3D &pt, const CellG *newCell,const CellG *oldCell,const unsigned int _flipAttempt);

	  

      virtual void setLastFlipAccepted(bool _accept);
	  virtual void set_aceptance_probability(double _prob);
	  
	  //virtual void request_current_mcs_accepted_mask_array(bool * mask_array, size_t len);
	  //virtual void  request_current_mcs_prob_array(double * double_array, size_t len);
	  //virtual void  request_current_mcs_accepted_mask_array(bool * boolvec, int n);
	  virtual void  get_current_mcs_accepted_mask_npy_array(int * intvec, int n);
	  virtual void  get_current_mcs_prob_npy_array(double * doublevec, int n);
	  virtual void  get_current_mcs_flip_attempt_points_npy_array(short * shortvec, int n);


	  virtual long get_number_energy_fcn_calculations();
	  virtual void range(int *rangevec, int n);

	  // Python reporting

	  std::vector<std::vector<double> > getCurrentEnergyChanges() { return totEnergyDataListCurrent; }
	  std::vector<bool> getCurrentFlipResults() { return accNotAccListCurrent; }

   private:
      
      int NTot;
      int NAcc;
      int NRej;
      int lastFlipAttempt;
	  long current_mcs_pos; // holds index of the totEnergyDataList's first item for the current mcs	  
//       unsigned int numberOfAttempts;
      std::vector<double> lastEnergyVec;
//       std::vector<double> totEnergyVector;
//       std::vector<double> accEnergyVector;
//       std::vector<double> rejEnergyVector;

	  std::list<Point3D> pixel_copy_attempt_points_list;
	  std::list<double> acceptance_probability_list;
	  std::list<int> mcs_list;

      std::list<std::vector<double> > totEnergyDataList; // sotres energies for each spin flip attempt
	  std::vector<std::vector<double> > totEnergyDataListCurrent; //stores energies for each spin flip attempt for current step; for reporting in Python
      std::list<bool> accNotAccList; //tells whether entry in totEnergyVecVec is accepted or not
	  std::vector<bool> accNotAccListCurrent; // tells whether entry in totEnergyDataListCurrent is accepted or not

      
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

/*      bool outputAcceptedSpinFlipHeader;
      bool outputRejectedSpinFlipHeader;
      bool outputTotalSpinFlipHeader;*/
      bool gatherResultsFilesPrepared;

      void writeHeader();
      void writeHeaderFlex(std::ofstream & _out);
      void writeDataLineFlex(std::ofstream & _out, std::vector<double> & _energies);
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

