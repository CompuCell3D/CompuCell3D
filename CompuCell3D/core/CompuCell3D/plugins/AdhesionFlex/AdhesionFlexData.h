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
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR1 PURPOSE.  See the GNU    *
*             General Public License for more details.                  *
*                                                                       *
*  You should have received a copy of the GNU General Public License    *
*     along with this program; if not, write to the Free Software       *
*      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
*************************************************************************/

#ifndef ADHESIONFLEXDATA_H
#define ADHESIONFLEXDATA_H

#include <CompuCell3D/CC3D.h>

// // // #include <vector>
// // // #include <set>
// // // #include <string>

#include "AdhesionFlexDLLSpecifier.h"
namespace CompuCell3D {

	class CellG;



	class ADHESIONFLEX_EXPORT AdhesionFlexData{

	public:
		typedef std::vector<float> ContainerType_t;
		AdhesionFlexData():adhesionMoleculeDensityVec(std::vector<float>(1,0.0)){}

		std::vector<float> adhesionMoleculeDensityVec;

		void assignValue(unsigned int _pos, float _value){
			if (_pos> adhesionMoleculeDensityVec.size()-1 ){
				unsigned int adhesionMoleculeDensityVecSize=adhesionMoleculeDensityVec.size();
				for(unsigned int i=0 ; i<_pos-(adhesionMoleculeDensityVecSize-1) ;++i){
					adhesionMoleculeDensityVec.push_back(0.);
				}
				adhesionMoleculeDensityVec[_pos]=_value;
			}else{
				adhesionMoleculeDensityVec[_pos]=_value;
			}
		}

		void assignAdhesionMoleculeDensityVector(std::vector<float> & _adhesionVector){
			adhesionMoleculeDensityVec=_adhesionVector;
		}     
		float getValue(unsigned int _pos){
			if (_pos>(adhesionMoleculeDensityVec.size()-1)){
				return 0.;
			} else{
				return adhesionMoleculeDensityVec[_pos];
			}  
		}
	};

	class ADHESIONFLEX_EXPORT AdhesionMoleculeDensityData{
	public:
		AdhesionMoleculeDensityData():density(0.0){}
		std::string molecule;
		float density;
	};


	//class ADHESIONFLEX_EXPORT CadherinData{
	//public:
	//	CadherinData(std::string _cad1Name,std::string _cad2Name,float _specificity):
	//	  cad1Name(_cad1Name),cad2Name(_cad2Name),specificity(_specificity)
	//	  {}
	//	  std::string cad1Name,cad2Name;
	//	  float specificity;
	//};

	//class ADHESIONFLEX_EXPORT AdhesionFlexSpecificityCadherin{
	//    public:
	//       AdhesionFlexSpecificityCadherin(){}
	//       
	//       std::set<std::string> cadherinNameLocalSet;

	//       std::vector<CadherinData> specificityCadherinTuppleVec;


	//       CadherinData * Specificity(std::string _cad1,std::string _cad2, double _spec){
	//          specificityCadherinTuppleVec.push_back(CadherinData(_cad1,_cad2,_spec));
	//          cadherinNameLocalSet.insert(_cad1);
	//          cadherinNameLocalSet.insert(_cad2);
	//          return &specificityCadherinTuppleVec[specificityCadherinTuppleVec.size()-1];
	//       }

	//       CadherinData * getSpecificity(std::string _cad1,std::string _cad2){
	//          for (int i=0 ; i < specificityCadherinTuppleVec.size() ; ++i){
	//             if(specificityCadherinTuppleVec[i].cad1Name==_cad1 && specificityCadherinTuppleVec[i].cad2Name==_cad2){
	//                return & specificityCadherinTuppleVec[i];
	//             }
	//             else if (specificityCadherinTuppleVec[i].cad1Name==_cad2 && specificityCadherinTuppleVec[i].cad2Name==_cad1){
	//                return & specificityCadherinTuppleVec[i];
	//             }
	//          }
	//          return 0;
	//       }



	// };










};
#endif
