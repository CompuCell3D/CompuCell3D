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

#ifndef ECMATERIALSDATA_H
#define ECMATERIALSDATA_H

#include <CompuCell3D/CC3D.h>

// // // #include <vector>
// // // #include <set>
// // // #include <string>

#include "ECMaterialsDLLSpecifier.h"
namespace CompuCell3D {

    class ECMATERIALS_EXPORT ECMaterialComponentData{
		public:
			ECMaterialComponentData() :durabilityLM(0), isTransferable(true) {}
			~ECMaterialComponentData() {}

			std::string ECMaterialName;
		    float durabilityLM;
		    bool isTransferable;

            void setName(std::string _name){ECMaterialName=_name;}
		    std::string getName(){return ECMaterialName;}
		    void setDurabilityLM(float _LM){durabilityLM=_LM;}
		    float getDurabilityLM(){return durabilityLM;}
		    void setTransferable(bool _isTransferable){isTransferable=_isTransferable;}
		    bool getTransferable(){return isTransferable;}

	};

    class ECMATERIALS_EXPORT ECMaterialsData{
		public:
            ECMaterialsData():numMtls((unsigned int) 1), ECMaterialsQuantityVec(std::vector<float>(0.0)){};
            ~ECMaterialsData(){}

			unsigned int numMtls;
			std::vector<float> ECMaterialsQuantityVec;

            void setECMaterialsQuantity(unsigned int _pos, float _qty){
                if(_pos>ECMaterialsQuantityVec.size()-1){return;}
                ECMaterialsQuantityVec[_pos]=_qty;
            }
            void setECMaterialsQuantityVec(std::vector<float> _qtyVec){ECMaterialsQuantityVec=_qtyVec;}
            float getECMaterialQuantity(unsigned int _pos){
                if(_pos>ECMaterialsQuantityVec.size()-1){return 0.0;}
                return ECMaterialsQuantityVec[_pos];
            }
            std::vector<float> getECMaterialsQuantityVec(){return ECMaterialsQuantityVec;}
            void setNewECMaterialsQuantityVec(unsigned int _numMtls){
				numMtls = _numMtls;
				std::vector<float> ECMaterialsQuantityVec(numMtls);
                ECMaterialsQuantityVec[0] = 1.0;
            }

	};

	class ECMATERIALS_EXPORT ECMaterialCellData{
	    public:
	        ECMaterialCellData(){};
	        ~ECMaterialCellData(){};

			std::vector<float> RemodelingQuantity;
			std::vector<float> AdhesionCoefficients;

	        void setNewRemodelingQuantityVec(unsigned int _numMtls){
	            std::vector<float> RemodelingQuantity(_numMtls, 0.0);
	            RemodelingQuantity[0] = 1.0;
	        }
	        void setRemodelingQuantityVec(std::vector<float> _qtyVec){std::vector<float> RemodelingQuantity(_qtyVec);}
	        void setRemodelingQuantity(unsigned int _pos, float _qty){
	            if(_pos>RemodelingQuantity.size()-1){return;}
                RemodelingQuantity[_pos]=_qty;
	        }
	        float getRemodelingQuantity(unsigned int _pos){
	            if(_pos>RemodelingQuantity.size()-1){return 0.0;}
                return RemodelingQuantity[_pos];
	        }
	        std::vector<float> getRemodelingQuantityVec(){return RemodelingQuantity;}

	        void setNewAdhesionCoefficientsVec(unsigned int _numMtls){std::vector<float> AdhesionCoefficients(_numMtls, 0.0);}
	        void setAdhesionCoefficientsVec(std::vector<float> _adhVec){std::vector<float> AdhesionCoefficients(_adhVec);}
	        void setAdhesionCoefficient(unsigned int _pos, float _adh){
	            if(_pos>AdhesionCoefficients.size()-1){return;}
                AdhesionCoefficients[_pos]=_adh;
	        }
	        float getAdhesionCoefficient(unsigned int _pos){
	            if(_pos>AdhesionCoefficients.size()-1){return 0.0;}
                return AdhesionCoefficients[_pos];
	        }
	        std::vector<float> getAdhesionCoefficientVec(){return AdhesionCoefficients;}
	};

};
#endif
