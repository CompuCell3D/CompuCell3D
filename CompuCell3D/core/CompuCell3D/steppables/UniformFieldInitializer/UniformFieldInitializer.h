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

#ifndef UNIFORMFIELDINITIALIZER_H
#define UNIFORMFIELDINITIALIZER_H
#include <CompuCell3D/CC3D.h>
#include "UniformFieldInitializerDLLSpecifier.h"

namespace CompuCell3D {
	class Potts3D;
	class Simulator;
	class CellInventory;

	class UNIFORMFIELDINITIALIZER_EXPORT UniformFieldInitializerData {

	public:
		UniformFieldInitializerData() :
			boxMin(Dim3D(0, 0, 0)), boxMax(Dim3D(0, 0, 0)), width(1), gap(0), randomize(false)
		{}
		Dim3D boxMin;
		Dim3D boxMax;
		std::vector<std::string> typeNames;
		std::string typeNamesString;
		int width;
		int gap;
		bool randomize;
		void BoxMin(Dim3D &_boxMin) { boxMin = _boxMin; }
		void BoxMax(Dim3D & _boxMax) { boxMax = _boxMax; }
		void Gap(int _gap) { gap = _gap; }
		void Width(int _width) { width = _width; }
		void Types(std::string  _type) {
			typeNames.push_back(_type);
		}
	};


	class UNIFORMFIELDINITIALIZER_EXPORT UniformFieldInitializer : public Steppable {
		Potts3D *potts;
		Simulator *sim;
		CellInventory * cellInventoryPtr;
		void layOutCells(const UniformFieldInitializerData & _initData);
		unsigned char initCellType(const UniformFieldInitializerData & _initData);
		std::vector<UniformFieldInitializerData> initDataVec;

	public:

		UniformFieldInitializer();
		virtual ~UniformFieldInitializer() {};
		void setPotts(Potts3D *potts) { this->potts = potts; }

		void initializeCellTypes();
		// SimObject interface
		virtual void init(Simulator *simulator, CC3DXMLElement * _xmlData = 0);

		// Begin Steppable interface
		virtual void start();
		virtual void step(const unsigned int currentStep) {}
		virtual void finish() {}
		// End Steppable interface
		virtual std::string steerableName();
		virtual std::string toString();



	};
};
#endif
