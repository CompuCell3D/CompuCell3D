#ifndef ENERGYFUNCTIONCALCLUATOR_H
#define ENERGYFUNCTIONCALCLUATOR_H
#include <string>
#include <vector>
#include <map>


class CC3DXMLElement;

namespace CompuCell3D{

class EnergyFunction;
class CellG;
class Point3D;
class Potts3D;
class Simulator;
class ParseData;

class EnergyFunctionCalculator{

   public:
      EnergyFunctionCalculator();
      virtual ~EnergyFunctionCalculator();

		virtual void init(CC3DXMLElement *_xmlData){}
      virtual void registerEnergyFunction(EnergyFunction *_function);
      virtual void registerEnergyFunctionWithName(EnergyFunction *_function,std::string _functionName);
      virtual void unregisterEnergyFunction(std::string _functionName);
      virtual void configureEnergyCalculator(std::vector<std::string> &_configVector){}
	  virtual long get_number_energy_fcn_calculations() { return 0; }
	  virtual void range(int *rangevec, int n) {}

      virtual double changeEnergy(Point3D &pt, const CellG *newCell,
				const CellG *oldCell,const unsigned int _flipAttempt);
      void setPotts(Potts3D * _potts){potts=_potts;}
      void setSimulator(Simulator * _sim){sim=_sim ;}
	  
	  //virtual void  request_current_mcs_accepted_mask_array(bool * mask_array, size_t len) {}
	  //virtual void  request_current_mcs_prob_array(double * double_array, size_t len) {}
	  //virtual void  request_current_mcs_accepted_mask_array(bool * boolvec, int n) {}
	  virtual void  get_current_mcs_accepted_mask_npy_array(int * intvec, int n) {}
	  virtual void  get_current_mcs_prob_npy_array(double * doublevec, int n) {}
	  virtual void  get_current_mcs_flip_attempt_points_npy_array(short * shortvec, int n) {}



      virtual void setLastFlipAccepted(bool _accept){lastFlipAccepted=_accept;}
	  virtual void set_aceptance_probability(double _prob) { }

	  // Python reporting

	  std::vector<std::string> getEnergyFunctionNames() { return energyFunctionsNameVec; }
	  virtual std::vector<std::vector<double> > getCurrentEnergyChanges() { return std::vector<std::vector<double> >(); }
	  virtual std::vector<bool> getCurrentFlipResults() { return std::vector<bool>(); }

   protected:
      std::vector<EnergyFunction *> energyFunctions;
      std::vector<std::string> energyFunctionsNameVec;

      std::map<std::string,EnergyFunction *> nameToEnergyFuctionMap;
      Potts3D *potts;
      Simulator *sim;

      bool lastFlipAccepted;

};

};


#endif

