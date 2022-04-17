

#ifndef MOLECULARCONTACTPLUGIN_H
#define MOLECULARCONTACTPLUGIN_H

#include <CompuCell3D/Plugin.h>

#include <CompuCell3D/Potts3D/EnergyFunction.h>
// #include <CompuCell3D/Potts3D/Cell.h>

//#include <CompuCell3D/dllDeclarationSpecifier.h>
#include "MolecularContactDLLSpecifier.h"

#include <BasicUtils/BasicClassAccessor.h>
#include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation
#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include "MolecularContactData.h"

class CC3DXMLElement;

namespace CompuCell3D {


  template <class T> class Field3D;
  template <class T> class WatchableField3D;

  class Point3D;
  class Simulator;
  class BoundaryStrategy;
  class MolecularContactDataContainer;
  class CellInventory;
  class Automaton;
  /**
   * Calculates surface energy based on a target surface and
   * lambda surface.
   */
  class BoundaryStrategy;
  class MOLECULARCONTACT_EXPORT MolecularContactPlugin : public Plugin, public CellGChangeWatcher,public EnergyFunction{
    Field3D<CellG *> *cellFieldG;

   //energy function data
    typedef std::map<std::string,float> MoleculeNameMap_t;
    MoleculeNameMap_t moleculeNameMap;
    std::map<std::string,float>::iterator iterMoleName;

    typedef std::map<int, std::string> contactEnergyEqns_t;
    typedef std::vector<std::vector<std::string> > contactEnergyEqnsArray_t;

    contactEnergyEqns_t contactEnergyEqns;

    contactEnergyEqnsArray_t contactEnergyEqnsArray;


    //energy value data values
    typedef std::map<int, double> contactEnergies_t;
    typedef std::vector<std::vector<double> > contactEnergyArray_t;

    contactEnergies_t contactEnergies;

    contactEnergyArray_t contactEnergyArray;


    Automaton *automaton;

    unsigned int maxNeighborIndex;
    double depth;
    Potts3D *potts;


    float targetLengthMolecularContact;
    float maxLengthMolecularContact;
    double lambdaMolecularContact;
    bool energyFuncFlag;
    Simulator *sim;
    Dim3D fieldDim;
    bool initialized;
    BoundaryStrategy  *boundaryStrategy;
    CellInventory * cellInventoryPtr;
    bool weightDistance;

//    BasicClassAccessor<ContactLocalFlexDataContainer> contactDataContainerAccessor;
   BasicClassAccessor<MolecularContactDataContainer> molecularContactDataAccessor;
   BasicClassAccessor<MolecularContactDataContainer> * molecularContactDataAccessorPtr;

  public:
    MolecularContactPlugin();
    virtual ~MolecularContactPlugin();


    BasicClassAccessor<MolecularContactDataContainer> * getMolecularContactDataAccessorPtr(){return & molecularContactDataAccessor;}

    //Plugin interface
    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
    virtual std::string toString();
    virtual void extraInit(Simulator *simulator);
    virtual void field3DChange(const Point3D &pt, CellG *newCell,CellG *oldCell);

   //EnergyFunction interface
   virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                const CellG *oldCell);


    //SteerableObject interface
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();
    void initializeMolecularConcentrations();
    void setConcentration(CellG * _cell, std::string MoleName, float val);
    void setContactEnergyEqn(const std::string typeName1,
                             const std::string typeName2, const std::string energyEqn);
    void setContactEnergy(const std::string typeName1,
                          const std::string typeName2, const double energy);
    int getIndex(const int type1, const int type2) const;
    double contactEnergyEqn(const CellG *cell1, const CellG *cell2);
     int mcsState;

    //EnergyFunction methods


  };
};
#endif
