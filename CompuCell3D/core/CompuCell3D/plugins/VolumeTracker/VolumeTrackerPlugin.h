

#ifndef VOLUMETRACKERPLUGIN_H
#define VOLUMETRACKERPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "VolumeTrackerDLLSpecifier.h"



class CC3DXMLElement;

namespace CompuCell3D {
  class Potts3D;
  class CellG;
  class Simulator;

  


  class VOLUMETRACKER_EXPORT VolumeTrackerPlugin : public Plugin, public CellGChangeWatcher, public Stepper 
  {
	Potts3D *potts;
	CellG *deadCellG;
	Simulator *sim;
	ParallelUtilsOpenMP *pUtils;
	ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;
    
	std::vector<CellG *> deadCellVec; 



  public:
	VolumeTrackerPlugin();
	virtual ~VolumeTrackerPlugin();
	
	void initVec(const vector<int> & _vec);
	void initVec(const Dim3D & _dim);

	// SimObject interface
	virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);
	
	virtual void handleEvent(CC3DEvent & _event);

	// CellChangeWatcher interface
	virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);
	bool checkIfOKToResize(Dim3D _newSize,Dim3D _shiftVec);
	// Stepper interface
	virtual void step();
	virtual std::string toString();
	virtual std::string steerableName();
  };
};
#endif
