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

	enum FocalPointPlasticityLinkType : unsigned char { REGULAR = 0, INTERNAL = 1, ANCHOR = 2 };

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

	public:

		FocalPointPlasticityLinkBase() :
			initiator(0), initiated(0), potts(0), fppltd(FocalPointPlasticityLinkTrackerData())
		{}
		~FocalPointPlasticityLinkBase() {
			delete initiator;
			delete initiated;
			delete potts;
			initiator = 0;
			initiated = 0;
			potts = 0;
		}

		FocalPointPlasticityLinkType getType() const { return type; }

		// Legacy support
		std::vector<FocalPointPlasticityTrackerData> getFPPTrackerData();

		// Derived properties

		// Length of link
		float getDistance();
		// Link tension
		float getTension();

		// General interface

		// Set the string for the constitutive law of this link
		void setConstitutiveLaw(std::string _lawString);
		// Return whether this link has a constitutive law
		bool hasLocalLaw() { return usingLocalLaw; }

		double constitutiveLaw(float _lambda, float _length, float _targetLength);

		// Data interface

		// Get lambda distance
		float getLambdaDistance() const { return fppltd.lambdaDistance; }
		// Set lambda distance
		void setLambdaDistance(float _lambdaDistance) { fppltd.lambdaDistance = _lambdaDistance; }
		// Get target distance
		float getTargetDistance() const { return fppltd.targetDistance; }
		// Set target distance
		void setTargetDistance(float _targetDistance) { fppltd.targetDistance = _targetDistance; }
		// Get maximum distance
		float getMaxDistance() const { return fppltd.maxDistance; }
		// Set maximum distance
		void setMaxDistance(float _maxDistance) { fppltd.maxDistance = _maxDistance; }
		// Get maximum number of junctions
		int getMaxNumberOfJunctions() const { return fppltd.maxNumberOfJunctions; }
		// Set maximum number of junctions
		void setMaxNumberOfJunctions(float _maxNumberOfJunctions) { fppltd.maxNumberOfJunctions = _maxNumberOfJunctions; }
		// Get activation energy
		float getActivationEnergy() const { return fppltd.activationEnergy; }
		// Set activation energy
		void setActivationEnergy(float _activationEnergy) { fppltd.activationEnergy = _activationEnergy; }
		// Get neighbor order
		int getNeighborOrder() const { return fppltd.neighborOrder; }
		// Set neighbor order
		void getNeighborOrder(int _neighborOrder) { fppltd.neighborOrder = _neighborOrder; }
		// Get is anchor
		bool isAnchor() const { return fppltd.anchor; }
		// Get initialization step
		int getInitMCS() const { return fppltd.initMCS; }
		// Get first id
		virtual long getId0() = 0;
		// Get second id
		virtual long getId1() = 0;

	};

	class FPPLinkInventory;

	class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityLink : public FocalPointPlasticityLinkBase {

		FocalPointPlasticityLinkType type = FocalPointPlasticityLinkType::REGULAR;

		friend FPPLinkInventory;

	public:
		FocalPointPlasticityLink() {};
		FocalPointPlasticityLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts, const FocalPointPlasticityLinkTrackerData &_fppltd)
		{
			initiator = _initiator;
			initiated = _initiated;
			potts = _potts;
			fppltd = FocalPointPlasticityLinkTrackerData(_fppltd);
			fppltd.anchor = false;
		}
		FocalPointPlasticityLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts, float _lambdaDistance = 0.0, float _targetDistance = 0.0, float _maxDistance = 100000.0, int _maxNumberOfJunctions = 0, float _activationEnergy = 0.0, int _neighborOrder = 1, int _initMCS = 0)
		{
			initiator = _initiator;
			initiated = _initiated;
			potts = _potts;
			fppltd = FocalPointPlasticityLinkTrackerData(_lambdaDistance, _targetDistance, _maxDistance, _maxNumberOfJunctions, _activationEnergy, _neighborOrder, _initMCS);
		}

		long getId0() const { return initiator->id; }
		long getId1() const { return initiated->id; }

	};

	class FPPInternalLinkInventory;

	class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityInternalLink : public FocalPointPlasticityLinkBase {

		FocalPointPlasticityLinkType type = FocalPointPlasticityLinkType::INTERNAL;

		friend FPPInternalLinkInventory;

	public:
		FocalPointPlasticityInternalLink() {};
		FocalPointPlasticityInternalLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts, const FocalPointPlasticityLinkTrackerData &_fppltd)
		{
			initiator = _initiator;
			initiated = _initiated;
			potts = _potts;
			fppltd = FocalPointPlasticityLinkTrackerData(_fppltd);
			fppltd.anchor = false;
		}
		FocalPointPlasticityInternalLink(CellG *_initiator, CellG *_initiated, Potts3D *_potts, float _lambdaDistance = 0.0, float _targetDistance = 0.0, float _maxDistance = 100000.0, int _maxNumberOfJunctions = 0, float _activationEnergy = 0.0, int _neighborOrder = 1, int _initMCS = 0)
		{
			initiator = _initiator;
			initiated = _initiated;
			potts = _potts;
			fppltd = FocalPointPlasticityLinkTrackerData(_lambdaDistance, _targetDistance, _maxDistance, _maxNumberOfJunctions, _activationEnergy, _neighborOrder, _initMCS);
		}

		long getId0() const { return initiator->id; }
		long getId1() const { return initiated->id; }

	};

	class FPPAnchorInventory;

	class FOCALPOINTPLASTICITY_EXPORT FocalPointPlasticityAnchor : public FocalPointPlasticityLinkBase {

		FocalPointPlasticityLinkType type = FocalPointPlasticityLinkType::ANCHOR;

		friend FPPAnchorInventory;

	public:
		FocalPointPlasticityAnchor() {}
		FocalPointPlasticityAnchor(CellG *_cell, Potts3D *_potts, const FocalPointPlasticityLinkTrackerData &_fppltd)
		{
			initiator = _cell;
			potts = _potts;
			fppltd = FocalPointPlasticityLinkTrackerData(_fppltd);
			fppltd.anchor = true;
			fppltd.anchorPoint = _fppltd.anchorPoint;
		}
		FocalPointPlasticityAnchor(CellG *_cell, Potts3D *_potts, float _lambdaDistance = 0.0, float _targetDistance = 0.0, float _maxDistance = 100000.0, int _maxNumberOfJunctions = 0, float _activationEnergy = 0.0, int _neighborOrder = 1, int _initMCS = 0, std::vector<float> _anchorPoint = std::vector<float>(3, 0.0)) {
			initiator = _cell;
			potts = _potts;
			fppltd = FocalPointPlasticityLinkTrackerData(_lambdaDistance, _targetDistance, _maxDistance, _maxNumberOfJunctions, _activationEnergy, _neighborOrder, _initMCS);
			fppltd.anchor = true;
			fppltd.anchorPoint = _anchorPoint;
		}

		long getId0() const { return initiator->id; }
		long getId1() const { return fppltd.anchorId; }

		// Get anchor point
		std::vector<float> getAnchorPoint() { return fppltd.anchorPoint; }
		// Set anchor point
		void getAnchorPoint(std::vector<float> _anchorPoint) { fppltd.anchorPoint = _anchorPoint; }

	};

}

#endif