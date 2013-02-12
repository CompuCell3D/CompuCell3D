#ifndef DIFFUSIONSOLVERFE_H
#define DIFFUSIONSOLVERFE_H

#include <CompuCell3D/CC3D.h>

// // // #include <CompuCell3D/Steppable.h>
// // // #include <CompuCell3D/Potts3D/Cell.h>
// // // #include <CompuCell3D/Boundary/BoundaryTypeDefinitions.h>

#include "DiffusableVectorCommon.h"

#include "DiffSecrData.h"

// // // #include <CompuCell3D/Serializer.h>

// // // #include <string>

// // // #include <vector>
// // // #include <set>
// // // #include <map>
// // // #include <iostream>

#include "PDESolversDLLSpecifier.h"

namespace CompuCell3D {

/**
@author m
*/
//forward declarations
class Potts3D;
class Simulator;
class Cell;
class CellInventory;
class Automaton;

class BoxWatcher;

class DiffusionData;
template <class Cruncher> class PDESOLVERS_EXPORT SecretionDataDiffusionFE;
template <class Cruncher> class PDESOLVERS_EXPORT DiffusionSolverSerializer;
class TestDiffusionSolver; // Testing DiffusionSolverFE
class ParallelUtilsOpenMP;
class CellTypeMonitorPlugin;

template <typename Y> class Field3D;
template <typename Y> class Field3DImpl;
template <typename Y> class WatchableField3D;


template <class Cruncher>
class PDESOLVERS_EXPORT DiffusionSolverFE;

template <class Cruncher>
class PDESOLVERS_EXPORT SecretionDataDiffusionFE:public SecretionData{
   public:
      typedef void (DiffusionSolverFE<Cruncher>::*secrSingleFieldFcnPtr_t)(unsigned int);
      std::vector<secrSingleFieldFcnPtr_t> secretionFcnPtrVec;
};

template <class Cruncher>
class PDESOLVERS_EXPORT DiffusionSecretionDiffusionFEFieldTupple{
   public:
      DiffusionData diffData;
      SecretionDataDiffusionFE<Cruncher> secrData;
      DiffusionData * getDiffusionData(){return &diffData;}
      SecretionDataDiffusionFE<Cruncher> * getSecretionData(){return &secrData;}
};

//CRT pattern is used to extract the common for CPU and GPU code part
//http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
template <class Cruncher>
class PDESOLVERS_EXPORT DiffusionSolverFE : public Steppable
{

  template <class CruncherFoo>
  friend class PDESOLVERS_EXPORT DiffusionSolverSerializer;

  // For Testing
  friend class TestDiffusionSolver; // In production version you need to enclose with #ifdef #endif

  public :
   typedef void (DiffusionSolverFE::*diffSecrFcnPtr_t)(void);
   typedef void (DiffusionSolverFE::*secrSingleFieldFcnPtr_t)(unsigned int);
   typedef float precision_t;
   //typedef Array3DBorders<precision_t>::ContainerType Array3D_t;
 //  typedef typename Cruncher::Array_t ConcentrationField_t;
//     typedef typename Cruncher::qq_t ConcentrationField_t;

	BoxWatcher *boxWatcherSteppable;
	
	float diffusionLatticeScalingFactor; // for hex in 2Dlattice it is 2/3.0 , for 3D is 1/2.0, for cartesian lattice it is 1
	bool autoscaleDiffusion;

protected:

   Potts3D *potts;
   Simulator *simPtr;
   ParallelUtilsOpenMP *pUtils;

   unsigned int currentStep;
   unsigned int maxDiffusionZ;
   float diffConst;
   
   float decayConst;
   float deltaX;///spacing
   float deltaT;///time interval
   float dt_dx2; ///ratio delta_t/delta_x^2
   WatchableField3D<CellG *> *cellFieldG;
   Automaton *automaton;

//    std::vector<DiffusionData> diffDataVec;
//    std::vector<SecretionDataDiffusionFE> secrDataVec;
   std::vector<bool> periodicBoundaryCheckVector;

   std::vector<BoundaryConditionSpecifier> bcSpecVec;
   std::vector<bool> bcSpecFlagVec;



   CellInventory *cellInventoryPtr;

   void (DiffusionSolverFE::*diffusePtr)(void);///ptr to member method - Forward Euler diffusion solver
   void (DiffusionSolverFE::*secretePtr)(void);///ptr to member method - Forward Euler diffusion solver

   void diffuse();

   void secrete();
   void secreteOnContact();

   void secreteSingleField(unsigned int idx);

   void secreteOnContactSingleField(unsigned int idx);

   void secreteConstantConcentrationSingleField(unsigned int idx);

   template <typename ConcentrationField_t>
   void scrarch2Concentration(ConcentrationField_t *concentrationField, ConcentrationField_t *scratchField);

   template <typename ConcentrationField_t>
   void outputField( std::ostream & _out, ConcentrationField_t *_concentrationField);

   template <typename ConcentrationField_t>
   void readConcentrationField(std::string fileName,ConcentrationField_t *concentrationField);
   //void boundaryConditionInit(ConcentrationField_t *concentrationField);
   
   void boundaryConditionInit(int idx);
   bool isBoudaryRegion(int x, int y, int z, Dim3D dim);

   unsigned int numberOfFields;
   Dim3D fieldDim;
   Dim3D workFieldDim;

   float couplingTerm(Point3D & _pt,std::vector<CouplingData> & _couplDataVec,float _currentConcentration);

   //template <typename ConcentrationField_t>
   void initializeConcentration();

   bool serializeFlag;
   bool readFromFileFlag;
   unsigned int serializeFrequency;

   DiffusionSolverSerializer<Cruncher> *serializerPtr;
   bool haveCouplingTerms;
   std::vector<DiffusionSecretionDiffusionFEFieldTupple<Cruncher> >  diffSecrFieldTuppleVec;
	//vector<string> concentrationFieldNameVectorTmp;

   int scalingExtraMCS;
   std::vector<int> scalingExtraMCSVec; //TODO: check if used
   std::vector<float> diffConstVec; 
   std::vector<float> decayConstVec; 

   CellTypeMonitorPlugin *cellTypeMonitorPlugin;
   Array3DCUDA<unsigned char> * h_celltype_field;

   std::vector<std::vector<Point3D> > hexOffsetArray;
   std::vector<Point3D> offsetVecCartesian;
   LatticeType latticeType;

   const std::vector<Point3D> & getOffsetVec(Point3D & pt) const {
      if(latticeType==HEXAGONAL_LATTICE){
         return hexOffsetArray[(pt.z%3)*2+pt.y%2];
      }else{
         return offsetVecCartesian;
      }
   }
   bool checkIfOffsetInArray(Point3D _pt, std::vector<Point3D> & _array);
   void prepareForwardDerivativeOffsets();


   //functions to realize in derived classes
   virtual void initImpl()=0;//first step of initialization
   virtual void extraInitImpl()=0;//second step of initialization, when more parameters are known
   virtual void initCellTypesAndBoundariesImpl()=0;

   virtual void solverSpecific(CC3DXMLElement *_xmlData)=0;//reading solver-specific information from XML file

   //template <typename ConcentrationField_t>
   //void diffuseSingleFieldImpl(ConcentrationField_t &concentrationField, DiffusionData &diffData)=0;

public:

    DiffusionSolverFE();

    virtual ~DiffusionSolverFE();


    virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData=0);
    virtual void extraInit(Simulator *simulator);

    // Begin Steppable interface
    virtual void start();
    virtual void step(const unsigned int _currentStep);
    virtual void finish() {}
    // End Steppable interface

    //SteerableObject interface
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();
	 virtual std::string toString();

	private:

		void diffuseSingleField(unsigned int idx);
		/*template <typename ConcentrationField_t>
		ConcentrationField_t const* getConcentrationField(int n)const;*/

};

template <class Cruncher>
class PDESOLVERS_EXPORT DiffusionSolverSerializer: public Serializer{
   public:
      DiffusionSolverSerializer():Serializer(){
         solverPtr=0;
         serializedFileExtension="dat";
         currentStep=0;
      }
      ~DiffusionSolverSerializer(){}
      DiffusionSolverFE<Cruncher> * solverPtr;
      virtual void serialize();
      virtual void readFromFile();
      void setCurrentStep(unsigned int _currentStep){currentStep=_currentStep;}
   protected:
   unsigned int currentStep;

};


};//namespace CompuCell3D



#endif
