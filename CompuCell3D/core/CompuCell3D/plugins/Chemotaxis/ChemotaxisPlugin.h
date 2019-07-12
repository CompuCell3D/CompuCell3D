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

#ifndef CHEMOTAXISPLUGIN_H
#define CHEMOTAXISPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "ChemotaxisData.h"
#include "ChemotaxisDLLSpecifier.h"


class CC3DXMLElement;
namespace CompuCell3D {
  
  class Simulator;

  class Automaton;

  template <class T>
  class Field3DImpl;

  template <class T>
  class Field3D;

  class Potts3D;
  //class ChemotaxisData;

  class CHEMOTAXIS_EXPORT ChemotaxisPlugin : public Plugin,public EnergyFunction {

    Simulator* sim;
	 CC3DXMLElement * xmlData;
	 Automaton *automaton;
	 //EnergyFunction data

    Potts3D *potts;
    std::vector<Field3D<float> *> fieldVec;
	std::vector<std::string> fieldNameVec;
	
   std::vector<std::vector<ChemotaxisData> > vecVecChemotaxisData;

   float simpleChemotaxisFormula(float _flipNeighborConc,float _conc,ChemotaxisData & _chemotaxisData);
   float saturationChemotaxisFormula(float _flipNeighborConc,float _conc,ChemotaxisData & _chemotaxisData);
   float saturationLinearChemotaxisFormula(float _flipNeighborConc,float _conc,ChemotaxisData & _chemotaxisData);
   float saturationDifferenceChemotaxisFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData);
   float powerChemotaxisFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData);
   float log10DivisionFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData);
   float logNatDivisionFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData);
   float log10DifferenceFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData);
   float logNatDifferenceFormula(float _flipNeighborConc, float _conc, ChemotaxisData & _chemotaxisData);
   
   //bool okToChemotact(unsigned int fieldIdx,unsigned char cellType);
   std::string chemotaxisAlgorithm;

   BasicClassAccessor< std::map<std::string,ChemotaxisData> > chemotaxisDataAccessor;
	 
  public:
    typedef float (ChemotaxisPlugin::*chemotaxisEnergyFormulaFcnPtr_t)(float,float,ChemotaxisData &);

    typedef double (ChemotaxisPlugin::*changeEnergyEnergyFormulaFcnPtr_t)(const Point3D &, const CellG *,
                                const CellG *);
   private:
      changeEnergyEnergyFormulaFcnPtr_t algorithmPtr;
	  std::map<std::string,chemotaxisEnergyFormulaFcnPtr_t> chemotaxisFormulaDict;


  public:
    ChemotaxisPlugin();
    virtual ~ChemotaxisPlugin();

    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);
    virtual void extraInit(Simulator *simulator);

	 //EnergyFunction interface    
    virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                const CellG *oldCell);



    //Steerable interface
    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);
    virtual std::string steerableName();
	 virtual std::string toString();



   
    
    //EnergyFunction Methods    

    
    double regularChemotaxis(const Point3D &pt, const CellG *newCell,
                                const CellG *oldCell);
   
    double merksChemotaxis(const Point3D &pt, const CellG *newCell,
                                const CellG *oldCell);

	ChemotaxisData * addChemotaxisData(CellG * _cell,std::string _fieldName);
	ChemotaxisData * getChemotaxisData(CellG * _cell,std::string _fieldName);

    //misslepped name - should be getFieldNamesWithChemotaxisData
	std::vector<std::string> getFieldNamesWithChemotaxisData(CellG * _cell);

	 

  };
};
#endif
