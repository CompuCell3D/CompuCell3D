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

#ifndef ADHESIONFLEXPLUGIN_H
#define ADHESIONFLEXPLUGIN_H

#include <CompuCell3D/CC3D.h>


#include "AdhesionFlexData.h"


// // // #include <CompuCell3D/Potts3D/EnergyFunction.h>
// // // #include <BasicUtils/BasicClassAccessor.h>
// // // #include <BasicUtils/BasicClassGroup.h> //had to include it to avoid problems with template instantiation

// // // #include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
// // // #include <PublicUtilities/ParallelUtilsOpenMP.h>
// // // #include <CompuCell3D/Plugin.h>
// // // #include <muParser/muParser.h>
#include "AdhesionFlexDLLSpecifier.h"


class CC3DXMLElement;

namespace CompuCell3D {
	class Simulator;

	class Potts3D;
	class Automaton;
	class AdhesionFlexData;
	class BoundaryStrategy;
    class ParallelUtilsOpenMP;


	class ADHESIONFLEX_EXPORT  AdhesionFlexPlugin : public Plugin,public EnergyFunction {
	public:
		typedef double (AdhesionFlexPlugin::*adhesionFlexEnergyPtr_t)(const CellG *cell1, const CellG *cell2);


	private:
		BasicClassAccessor<AdhesionFlexData> adhesionFlexDataAccessor;
		CC3DXMLElement *xmlData;
		Potts3D *potts;
		Simulator *sim;
	    ParallelUtilsOpenMP *pUtils;            
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;        
		//Energy function data


		typedef std::map<int, double> bindingParameters_t;

		typedef std::vector<std::vector<double> > bindingParameterArray_t;

		
		std::map<std::string,unsigned int> mapCadNameToIndex;
		unsigned int numberOfCadherins;
		//     multiSpecificityArray_t multiSpecificityArray;

		std::string contactFunctionType;
		std::string autoName;
		double depth;

		

		BasicClassAccessor<AdhesionFlexData> * adhesionFlexDataAccessorPtr;

		Automaton *automaton;
		bool weightDistance;

		adhesionFlexEnergyPtr_t adhesionFlexEnergyPtr;

		unsigned int maxNeighborIndex;
		BoundaryStrategy *boundaryStrategy;
		// float energyOffset;


		bindingParameters_t bindingParameters;
		bindingParameterArray_t bindingParameterArray;
		int numberOfAdhesionMolecules;
		bool adhesionDensityInitialized;
		std::vector<std::string> adhesionMoleculeNameVec;
		std::map<std::string, int> moleculeNameIndexMap;
		std::map<int,std::vector<float> > typeToAdhesionMoleculeDensityMap; 
		std::vector<float>  adhesionMoleculeDensityVecMedium;


		// std::string formulaString; //expression for cad-molecule adhesion function
		// double molecule1; //used to keep arguments for molecule-molecule adhesion function 
		// double molecule2;//used to keep arguments for cad-molecule adhesion function
		// mu::Parser p;
        
        //vectorized variables for convenient parallel access
        std::string formulaString; //expression for cad-molecule adhesion function
		std::vector<double> molecule1Vec; //used to keep arguments for molecule-molecule adhesion function 
		std::vector<double> molecule2Vec;//used to keep arguments for cad-molecule adhesion function
		std::vector<mu::Parser> pVec;
        
        
		//default non existing density
		static const int errorDensity=-1000000;
		void initializeAdhesionMoleculeDensityVector();

	public:

		AdhesionFlexPlugin();
		virtual ~AdhesionFlexPlugin();

		BasicClassAccessor<AdhesionFlexData> * getAdhesionFlexDataAccessorPtr(){return & adhesionFlexDataAccessor;}


		virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

		virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);


		virtual void extraInit(Simulator *simulator);
        
        virtual void handleEvent(CC3DEvent & _event);




		//Steerrable interface
		virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
		virtual std::string steerableName();
		virtual std::string toString();

		/**
		* @return The contact energy between cell1 and cell2.
		*/
		double adhesionFlexEnergyCustom(const CellG *cell1, const CellG *cell2);

		void setBindingParameter(const std::string moleculeName1, const std::string moleculeName2, const double parameter, bool parsing_flag=false) ;     
		void setBindingParameterDirect(const std::string moleculeName1, const std::string moleculeName2, const double parameter) ;
		void setBindingParameterByIndexDirect(int _idx1, int _idx2, const double parameter) ;
		
		std::vector<std::vector<double> > getBindingParameterArray();
		std::vector<std::string> getAdhesionMoleculeNameVec();
		
		//functions used to manipulate densities of adhesion molecules
		void setAdhesionMoleculeDensity(CellG * _cell, std::string _moleculeName, float _density);
		void setAdhesionMoleculeDensityByIndex(CellG * _cell, int _idx, float _density);
		void setAdhesionMoleculeDensityVector(CellG * _cell, std::vector<float> _denVec);
		void assignNewAdhesionMoleculeDensityVector(CellG * _cell, std::vector<float> _denVec);
		//Medium functions
		void setMediumAdhesionMoleculeDensity(std::string _moleculeName, float _density);
		void setMediumAdhesionMoleculeDensityByIndex( int _idx, float _density);
		void setMediumAdhesionMoleculeDensityVector(std::vector<float> _denVec);
		void assignNewMediumAdhesionMoleculeDensityVector(std::vector<float> _denVec);

		
		float getAdhesionMoleculeDensity(CellG * _cell, std::string _moleculeName);
		float getAdhesionMoleculeDensityByIndex(CellG * _cell, int _idx);
		std::vector<float> getAdhesionMoleculeDensityVector(CellG * _cell);
		//Medium functions
		float getMediumAdhesionMoleculeDensity(std::string _moleculeName);
		float getMediumAdhesionMoleculeDensityByIndex(int _idx);
		std::vector<float> getMediumAdhesionMoleculeDensityVector();
		void overrideInitialization();


	protected:
		/**
		* @return The index used for ordering contact energies in the map.
		*/
		int getIndex(const int type1, const int type2) const;


	};
};
#endif
