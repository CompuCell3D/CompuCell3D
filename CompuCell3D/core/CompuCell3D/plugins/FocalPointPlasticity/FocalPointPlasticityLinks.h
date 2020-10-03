/*************************************************************************
*    CompuCell - A software framework for multimodel simulations of     *
* biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
*                             Indiana                                   *
*                                                                       *
* This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
*  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
*  CompuCell GNU General Public License RIDER you can redistribute it   *
* and/or modify it under the terms of the GNU General Public License as *
*  published by the Free Software Foundation; either version 2 of the   *
*         License, or (at your option) any later version.               *
*                                                                       *
* This program is distributed in the hope that it will be useful, but   *
*      WITHOUT ANY WARRANTY; without even the implied warranty of       *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
*             General Public License for more details.                  *
*                                                                       *
*  You should have received a copy of the GNU General Public License    *
*     along with this program; if not, write to the Free Software       *
*      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
*************************************************************************/

#ifndef FOCALPOINTPLASTICITYLINKS_H
#define FOCALPOINTPLASTICITYLINKS_H

#include <CompuCell3D/CC3D.h>

#include "FocalPointPlasticityTracker.h"
#include "FocalPointPlasticityDLLSpecifier.h"

class ExpressionEvaluator;

namespace CompuCell3D {

	/**
	Written by T.J. Sego, Ph.D.
	*/

	class BoundaryStrategy;
	class CellG;
	class Potts3D;

	class FocalPointPlasticityTrackerData;
	class FocalPointPlasticityLinkTrackerData;

	// Link definitions

	enum FocalPointPlasticityLinkType { REGULAR, INTERNAL, ANCHOR };

	class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityLinkBase {

	private:

		FocalPointPlasticityLinkType type;

	protected:

		Potts3D *potts;

		CellG *initiator;
		CellG *initiated;
		FocalPointPlasticityLinkTrackerData fppltd;

		ExpressionEvaluator ev;
		void initializeConstitutiveLaw(std::string _localLaw);
		bool usingLocalLaw = false;

		CellG* getCellGFromConst(const CellG* _cell);

	public:

		FocalPointPlasticityLinkBase() :
			initiator(0), initiated(0), potts(0), fppltd(FocalPointPlasticityLinkTrackerData())
		{}
		~FocalPointPlasticityLinkBase() {}

		const FocalPointPlasticityLinkType getType() { return type; }

		// Legacy support
		FocalPointPlasticityTrackerData getFPPTrackerData(CellG* _cell) {
			FocalPointPlasticityTrackerData fpptd = FocalPointPlasticityTrackerData(fppltd);
			fpptd.neighborAddress = getOtherCell(_cell);
			fpptd.isInitiator = isInitiator(_cell);
			fpptd.maxNumberOfJunctions = getMaxNumberOfJunctions();
			fpptd.activationEnergy = getActivationEnergy();
			fpptd.neighborOrder = getNeighborOrder();
			return fpptd;
		}
		FocalPointPlasticityTrackerData getFPPTrackerData(const CellG* _cell) { return getFPPTrackerData(getCellGFromConst(_cell)); }

		// Derived properties

		// Length of link
		float getDistance();
		// Link tension
		float getTension();

		// General interface

		// Set the string for the constitutive law of this link
		void setConstitutiveLaw(std::string _lawString);
		// Return whether this link has a constitutive law
		const bool hasLocalLaw() { return usingLocalLaw; }

		// Pass the other cell of this link
		CellG* getOtherCell(CellG* _cell) {
			if (_cell) {
				if (initiator && _cell->id == initiator->id) return initiated;
				if (initiated && _cell->id == initiated->id) return initiator;
			}
			else {
				if (!initiator) return initiated;
				if (!initiated) return initiator;
			}
			ASSERT_OR_THROW("Cell is not a member of this link", false)
		}
		CellG* getOtherCell(const CellG* _cell) { return getOtherCell(const_cast<CellG*>(_cell)); }
		// Pass whether this cell is the initiator
		bool isInitiator(CellG* _cell) {
			if (_cell) {
				if (initiator && _cell->id == initiator->id) return false;
				if (initiated && _cell->id == initiated->id) return true;
			}
			else {
				if (!initiator) return false;
				if (!initiated) return true;
			}
			ASSERT_OR_THROW("Cell is not a member of this link", false)
		}

		double constitutiveLaw(float _lambda, float _length, float _targetLength);

		// Data interface

		// Get lambda distance
		const float getLambdaDistance() { return fppltd.lambdaDistance; }
		// Set lambda distance
		void setLambdaDistance(float _lambdaDistance) { fppltd.lambdaDistance = _lambdaDistance; }
		// Get target distance
		const float getTargetDistance() { return fppltd.targetDistance; }
		// Set target distance
		void setTargetDistance(float _targetDistance) { fppltd.targetDistance = _targetDistance; }
		// Get maximum distance
		const float getMaxDistance() { return fppltd.maxDistance; }
		// Set maximum distance
		void setMaxDistance(float _maxDistance) { fppltd.maxDistance = _maxDistance; }
		// Get maximum number of junctions
		const int getMaxNumberOfJunctions() { return fppltd.maxNumberOfJunctions; }
		// Set maximum number of junctions
		void setMaxNumberOfJunctions(int _maxNumberOfJunctions) { fppltd.maxNumberOfJunctions = _maxNumberOfJunctions; }
		// Get activation energy
		const float getActivationEnergy() { return fppltd.activationEnergy; }
		// Set activation energy
		void setActivationEnergy(float _activationEnergy) { fppltd.activationEnergy = _activationEnergy; }
		// Get neighbor order
		const int getNeighborOrder() { return fppltd.neighborOrder; }
		// Set neighbor order
		void setNeighborOrder(int _neighborOrder) { fppltd.neighborOrder = _neighborOrder; }
		// Get is anchor
		const bool isAnchor() { return fppltd.anchor; }
		// Get initialization step
		const int getInitMCS() { return fppltd.initMCS; }
		// Get first id
		virtual const long getId0() { return long(0); };
		// Get second id
		virtual const long getId1() { return long(0); };
		// Get first object
		CellG* getObj0() { return initiator; }
		// Get second object
		CellG* getObj1() { return initiated; }
	};

	class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityLink : public FocalPointPlasticityLinkBase {

		FocalPointPlasticityLinkType type = FocalPointPlasticityLinkType::REGULAR;

	public:
		FocalPointPlasticityLink() {};
		FocalPointPlasticityLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts, FocalPointPlasticityLinkTrackerData _fppltd)
		{
			initiator = _initiator;
			initiated = _initiated;
			potts = _potts;
			fppltd = FocalPointPlasticityLinkTrackerData(_fppltd);
			fppltd.anchor = false;
		}
		FocalPointPlasticityLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts, FocalPointPlasticityTrackerData _fpptd) :
			FocalPointPlasticityLink(_initiator, _initiated, _potts, FocalPointPlasticityLinkTrackerData(_fpptd)) {}
		FocalPointPlasticityLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts, float _lambdaDistance = 0.0, float _targetDistance = 0.0, float _maxDistance = 100000.0, int _initMCS = 0) :
			FocalPointPlasticityLink(_initiator, _initiated, _potts, FocalPointPlasticityLinkTrackerData(_lambdaDistance, _targetDistance, _maxDistance, _initMCS))
		{}

		const long getId0() { return initiator->id; }
		const long getId1() { return initiated->id; }

	};

	class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityInternalLink : public FocalPointPlasticityLinkBase {

		FocalPointPlasticityLinkType type = FocalPointPlasticityLinkType::INTERNAL;

	public:
		FocalPointPlasticityInternalLink() {};
		FocalPointPlasticityInternalLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts, FocalPointPlasticityLinkTrackerData _fppltd)
		{
			initiator = _initiator;
			initiated = _initiated;
			potts = _potts;
			fppltd = FocalPointPlasticityLinkTrackerData(_fppltd);
			fppltd.anchor = false;
		}
		FocalPointPlasticityInternalLink(const CellG *_initiator, CellG *_initiated, Potts3D *_potts, FocalPointPlasticityLinkTrackerData _fppltd) :
			FocalPointPlasticityInternalLink(getCellGFromConst(_initiator), _initiated, _potts, _fppltd) {}
		FocalPointPlasticityInternalLink(CellG *_initiator, const CellG *_initiated, Potts3D *_potts, FocalPointPlasticityLinkTrackerData _fppltd) :
			FocalPointPlasticityInternalLink(_initiator, getCellGFromConst(_initiated), _potts, _fppltd) {}
		FocalPointPlasticityInternalLink(const CellG *_initiator, const CellG *_initiated, Potts3D *_potts, FocalPointPlasticityLinkTrackerData _fppltd) :
			FocalPointPlasticityInternalLink(getCellGFromConst(_initiator), getCellGFromConst(_initiated), _potts, _fppltd) {}

		FocalPointPlasticityInternalLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts, FocalPointPlasticityTrackerData _fpptd) :
			FocalPointPlasticityInternalLink(_initiator, _initiated, _potts, FocalPointPlasticityLinkTrackerData(_fpptd)) {}
		FocalPointPlasticityInternalLink(const CellG *_initiator, CellG *_initiated, Potts3D *_potts, FocalPointPlasticityTrackerData _fpptd) :
			FocalPointPlasticityInternalLink(getCellGFromConst(_initiator), _initiated, _potts, _fpptd) {}
		FocalPointPlasticityInternalLink(CellG *_initiator, const CellG *_initiated, Potts3D *_potts, FocalPointPlasticityTrackerData _fpptd) :
			FocalPointPlasticityInternalLink(_initiator, getCellGFromConst(_initiated), _potts, _fpptd) {}
		FocalPointPlasticityInternalLink(const CellG *_initiator, const CellG *_initiated, Potts3D *_potts, FocalPointPlasticityTrackerData _fpptd) :
			FocalPointPlasticityInternalLink(getCellGFromConst(_initiator), getCellGFromConst(_initiated), _potts, _fpptd) {}

		FocalPointPlasticityInternalLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts, float _lambdaDistance = 0.0, float _targetDistance = 0.0, float _maxDistance = 100000.0, int _initMCS = 0) :
			FocalPointPlasticityInternalLink(_initiator, _initiated, _potts, FocalPointPlasticityLinkTrackerData(_lambdaDistance, _targetDistance, _maxDistance, _initMCS)) {}
		FocalPointPlasticityInternalLink(const CellG *_initiator, CellG *_initiated, Potts3D *_potts, float _lambdaDistance = 0.0, float _targetDistance = 0.0, float _maxDistance = 100000.0, int _initMCS = 0) :
			FocalPointPlasticityInternalLink(getCellGFromConst(_initiator), _initiated, _potts, _lambdaDistance, _targetDistance, _maxDistance, _initMCS) {}
		FocalPointPlasticityInternalLink(CellG *_initiator, const CellG *_initiated, Potts3D *_potts, float _lambdaDistance = 0.0, float _targetDistance = 0.0, float _maxDistance = 100000.0, int _initMCS = 0) :
			FocalPointPlasticityInternalLink(_initiator, getCellGFromConst(_initiated), _potts, _lambdaDistance, _targetDistance, _maxDistance, _initMCS) {}
		FocalPointPlasticityInternalLink(const CellG *_initiator, const CellG *_initiated, Potts3D *_potts, float _lambdaDistance = 0.0, float _targetDistance = 0.0, float _maxDistance = 100000.0, int _initMCS = 0) :
			FocalPointPlasticityInternalLink(getCellGFromConst(_initiator), getCellGFromConst(_initiated), _potts, _lambdaDistance, _targetDistance, _maxDistance, _initMCS) {}

		const long getId0() { return initiator->id; }
		const long getId1() { return initiated->id; }

	};

	class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityAnchor : public FocalPointPlasticityLinkBase {

		FocalPointPlasticityLinkType type = FocalPointPlasticityLinkType::ANCHOR;

	public:
		FocalPointPlasticityAnchor() {}
		FocalPointPlasticityAnchor(CellG *_cell, Potts3D *_potts, FocalPointPlasticityLinkTrackerData _fppltd)
		{
			initiator = _cell;
			initiated = (CellG*)(0);
			potts = _potts;
			fppltd = FocalPointPlasticityLinkTrackerData(_fppltd);
			fppltd.anchor = true;
			fppltd.anchorPoint = _fppltd.anchorPoint;
		}
		FocalPointPlasticityAnchor(const CellG *_cell, Potts3D *_potts, FocalPointPlasticityLinkTrackerData _fppltd) :
			FocalPointPlasticityAnchor(getCellGFromConst(_cell), _potts, _fppltd) {}
		FocalPointPlasticityAnchor(CellG *_cell, Potts3D *_potts, FocalPointPlasticityTrackerData _fpptd) :
			FocalPointPlasticityAnchor(_cell, _potts, FocalPointPlasticityLinkTrackerData(_fpptd)) {}
		FocalPointPlasticityAnchor(const CellG *_cell, Potts3D *_potts, FocalPointPlasticityTrackerData _fpptd) :
			FocalPointPlasticityAnchor(getCellGFromConst(_cell), _potts, _fpptd) {}
		FocalPointPlasticityAnchor(CellG *_cell, Potts3D *_potts, float _lambdaDistance = 0.0, float _targetDistance = 0.0, float _maxDistance = 100000.0, int _initMCS = 0, std::vector<float> _anchorPoint = std::vector<float>(3, 0.0)) :
			FocalPointPlasticityAnchor(_cell, _potts, FocalPointPlasticityLinkTrackerData(_lambdaDistance, _targetDistance, _maxDistance, _initMCS)) {}
		FocalPointPlasticityAnchor(const CellG *_cell, Potts3D *_potts, float _lambdaDistance = 0.0, float _targetDistance = 0.0, float _maxDistance = 100000.0, int _initMCS = 0, std::vector<float> _anchorPoint = std::vector<float>(3, 0.0)) :
			FocalPointPlasticityAnchor(getCellGFromConst(_cell), _potts, _lambdaDistance, _targetDistance, _maxDistance, _initMCS, _anchorPoint) {}

		const long getId0() { return initiator->id; }
		const long getId1() { return fppltd.anchorId; }

		// Get anchor point
		std::vector<float> getAnchorPoint() { return fppltd.anchorPoint; }
		// Set anchor point
		void setAnchorPoint(std::vector<float> _anchorPoint) { fppltd.anchorPoint = _anchorPoint; }
		// Get anchor id
		const int getAnchorId() { return fppltd.anchorId; }

	};

}

#endif