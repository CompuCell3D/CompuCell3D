#ifndef PERSISTENCEPLUGIN_H
#define PERSISTENCEPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "Persistence.h"
#include "PersistenceDLLSpecifier.h"

#include <unordered_map>
#include <unordered_set>

class CC3DXMLElement;

namespace CompuCell3D {

	class Cell;
	class PersistenceModel;
  
    /**
    Written by T.J. Sego, Ph.D.
    */

    /**
     * @brief Persistent cell motility models
     */
	class PERSISTENCE_EXPORT PersistencePlugin : public Plugin, public EnergyFunction, public Steppable, public CellInventoryWatcher {

		Dim3D fieldDim;
		ExtraMembersGroupAccessor<PersistenceData> persistenceDataAccessor;
		Simulator *simulator;
		CC3DXMLElement *xmlData;

		unsigned int randomSeed;
		std::mt19937 prng;

		std::unordered_map<long, PersistenceModel*> modelInventory;
		std::unordered_map<unsigned char, CC3DXMLElement*> cellTypeModelElements;

		PersistenceModel* _maybePersistenceModelConstruct(CellG* cell);
		std::unordered_map<unsigned char, CC3DXMLElement*> _pullTypeModelElements(CC3DXMLElement* _xmlData);
		void _applyTypeModelElements();
		// Handles when cells are created while types are unavailable
		std::unordered_set<long> _toInitialize;

		void _recordToInitialize(const long& _id);
		void _recordInitialized(const long& _id);
    
	public:

		friend PersistenceModel;

		enum ForceMode {
            FORCETYPE_NONE = 0,
            FORCETYPE_EXTENSION = 1 << 0,
            FORCETYPE_RETRACTION = 1 << 1,
            FORCETYPE_RECIPROCAL = FORCETYPE_EXTENSION | FORCETYPE_RETRACTION
        };
		enum DisplacementType {
			DISPLACEMENTTYPE_REGULAR = 0,
			DISPLACEMENTTYPE_NORMALIZED = 1,
			DISPLACEMENTTYPE_MASS = 2
		};

		PersistencePlugin();
		virtual ~PersistencePlugin();

		ExtraMembersGroupAccessor<PersistenceData>* getPersistenceDataAccessorPtr(){return &persistenceDataAccessor;}

		// Plugin interface
		virtual void init(Simulator* _simulator, CC3DXMLElement* _xmlData=0);
		virtual void extraInit(Simulator* simulator);
		virtual std::string toString() { return "Persistence"; };
		virtual void handleEvent(CC3DEvent& _event) {};

		// Steerable interface
        virtual void update(CC3DXMLElement* _xmlData, bool _fullInitFlag = false);
        virtual std::string steerableName() { return toString(); };

		// EnergyFunction interface
		virtual double changeEnergy(const Point3D& pt, const CellG* newCell, const CellG* oldCell);

		// Steppable interface
        virtual void step(const unsigned int currentStep);

		// CellInventoryWatcher interface
		virtual void onCellAdd(CellG* cell);
		virtual void onCellRemove(CellG* cell);

		// User API

		const unsigned int getRandomSeed() { return randomSeed; }
		PersistenceData* getPersistenceData(CellG* cell);
		PersistenceModel* getModel(CellG* cell);
	};

	// Persistence models

	class PERSISTENCE_EXPORT PersistenceModel {
	
	private:

		Coordinates3D<double> _calculateDisplacement(const Point3D& pt, const Coordinates3D<double>& copyVector, const bool& isNewCell);
	
	protected:

		Simulator* simulator;
		PersistencePlugin* plugin;
		CellG* cell;
		PersistencePlugin::ForceMode forceMode;
		PersistencePlugin::DisplacementType dispType;

		Dim3D getFieldDim() { return plugin->fieldDim; }
		std::mt19937& getPRNG() { return plugin->prng; };
	
	public:

		PersistenceModel(Simulator* _simulator, PersistencePlugin* _plugin, CellG* _cell) : 
			simulator(_simulator),
			plugin(_plugin),
			cell(_cell)
		{};

		double changeEnergy(const Point3D& pt, const Coordinates3D<double>& copyVector, const bool& isNewCell);

		const Coordinates3D<double> directionVector();

		const Coordinates3D<double> persistenceVector();

		virtual const std::string modelName() = 0;

		virtual void applyModelData(CC3DXMLElement* xmlData, const bool& initialize);

		virtual void update() = 0;

	};

	class PERSISTENCE_EXPORT SRPersistenceModel : public PersistenceModel {

		std::vector<Coordinates3D<double> > comHist;
		size_t comIdx;
		bool initialized;
		bool looped;
	
	public:

		SRPersistenceModel(Simulator* _simulator, PersistencePlugin* _plugin, CellG* _cell) : 
			PersistenceModel(_simulator, _plugin, _cell),
			comHist{},
			comIdx{0},
			initialized{false},
			looped{false}
		{}
		
		virtual const std::string modelName() { return "Self-Reinforcing"; }

		virtual void applyModelData(CC3DXMLElement* xmlData, const bool& initialize);

		virtual void update();
	};

	class PERSISTENCE_EXPORT ANPersistenceModel : public PersistenceModel {

		double sqrtPeriod;
		double stDev1;
		double stDev2;
		double stDev3;
	
	public:

		ANPersistenceModel(Simulator* _simulator, PersistencePlugin* _plugin, CellG* _cell) : 
			PersistenceModel(_simulator, _plugin, _cell),
			sqrtPeriod{1.0},
			stDev1{0.0},
			stDev2{0.0},
			stDev3{0.0}
		{}
		
		virtual const std::string modelName() { return "AngularNoise"; }

		virtual void applyModelData(CC3DXMLElement* xmlData, const bool& initialize);

		virtual void update();
	};

};
#endif