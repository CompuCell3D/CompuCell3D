

#ifndef PIXELTRACKERPLUGIN_H
#define PIXELTRACKERPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "PixelTracker.h"
#include "PixelTrackerDLLSpecifier.h"


class CC3DXMLElement;
namespace CompuCell3D {

  class Cell;
  class Field3DIndex;
  template <class T> class Field3D;
  template <class T> class WatchableField3D;
  
class PIXELTRACKER_EXPORT PixelTrackerPlugin : public Plugin, public CellGChangeWatcher {


      Dim3D fieldDim;
      ExtraMembersGroupAccessor<PixelTracker> pixelTrackerAccessor;
      Simulator *simulator;
	  Potts3D *potts;
	  ParallelUtilsOpenMP *pUtils;
	  ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

	  std::vector<std::set<PixelTrackerData> > mediumPixelSet;
	  std::vector<std::vector<pair<Dim3D, Dim3D> > > sectionDimsVec;
	  bool trackMedium;
	  unsigned int getParitionNumber(const Point3D &_pt, unsigned int _workerNum=0);

	  bool fullInitAtStart;
	  bool fullInitState;
    
   public:
      PixelTrackerPlugin();
      virtual ~PixelTrackerPlugin();
      
      
      // Field3DChangeWatcher interface
      virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                 CellG *oldCell);
      
		//Plugin interface 
		virtual void init(Simulator *_simulator, CC3DXMLElement *_xmlData=0);
		virtual void extraInit(Simulator *simulator);
		virtual std::string toString();
		virtual void handleEvent(CC3DEvent & _event);		

		ExtraMembersGroupAccessor<PixelTracker> * getPixelTrackerAccessorPtr(){return & pixelTrackerAccessor;}
		//had to include this function to get set itereation working properly with Python , and Player that has restart capabilities
		PixelTrackerData * getPixelTrackerData(PixelTrackerData * _psd){return _psd;}

		virtual void enableMediumTracker(bool _trackMedium=true) { trackMedium = _trackMedium; }
		virtual void mediumTrackerDataInit();
		virtual bool trackingMedium() { return trackMedium; }
		virtual std::set<PixelTrackerData> getMediumPixelSet();
		// Thread-safe
		std::vector<std::set<PixelTrackerData> > getPixelWorkerSets() { return mediumPixelSet; }

		void enableFullInitAtStart(bool _fullInitAtStart = true) { fullInitAtStart = _fullInitAtStart; }
		bool fullyInitialized() { return fullInitState; }
		void fullTrackerDataInit(Point3D _ptChange = Point3D(-1, -1, -1), CellG *oldCell = 0);
  };

};
#endif
