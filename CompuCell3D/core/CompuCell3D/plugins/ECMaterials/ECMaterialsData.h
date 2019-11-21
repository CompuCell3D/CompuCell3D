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

	/**
	@author T.J. Sego, Ph.D.
	*/

    class ECMATERIALS_EXPORT ECMaterialComponentData{
			std::string ECMaterialName;
			float durabilityLM;
			bool isTransferable;

			std::map<std::string, float> toFieldReactionCoefficients; // source term for scalar field
			std::map<std::string, float> fromFieldReactionCoefficients; // source term for quantity
			std::map<std::string, std::vector<float> > materialReactionCoefficients;
			float materialDiffusionCoefficient;
			std::map<std::string, float> CellTypeCoefficientsProliferation;
			std::map<std::string, std::map<std::string, float> > CellTypeCoefficientsDifferentiation;
			std::map<std::string, float> CellTypeCoefficientsDeath;
			
		public:
			ECMaterialComponentData() :durabilityLM(0), isTransferable(true) {}
			~ECMaterialComponentData() {}

			void setName(std::string _name){ECMaterialName=_name;}
		    std::string getName(){return ECMaterialName;}
		    void setDurabilityLM(float _LM){durabilityLM=_LM;}
		    float getDurabilityLM(){return durabilityLM;}
		    void setTransferable(bool _isTransferable){isTransferable=_isTransferable;}
		    bool getTransferable(){return isTransferable;}
			
			// Sets a field reaction for field name _fieldName with coefficient _val
			// Registers field reaction if not previously registered
			void setToFieldReactionCoefficientByName(std::string _fieldName, float _val) {
				std::map<std::string, float>::iterator mitr = toFieldReactionCoefficients.find(_fieldName);
				if (mitr == toFieldReactionCoefficients.end()){
					cerr << "Registering field reaction coefficient for pair (" << ECMaterialName << ", " << _fieldName << "): " << _val << endl;
					toFieldReactionCoefficients.insert(make_pair(_fieldName, _val));
				}
				else toFieldReactionCoefficients[_fieldName] = _val;
			}
			// Sets a field reaction for field name _fieldName with coefficient _val
			// Registers field reaction if not previously registered
			void setFromFieldReactionCoefficientByName(std::string _fieldName, float _val) {
				std::map<std::string, float>::iterator mitr = fromFieldReactionCoefficients.find(_fieldName);
				if (mitr == fromFieldReactionCoefficients.end()) {
					cerr << "Registering field reaction coefficient for pair (" << ECMaterialName << ", " << _fieldName << "): " << _val << endl;
					fromFieldReactionCoefficients.insert(make_pair(_fieldName, _val));
				}
				else fromFieldReactionCoefficients[_fieldName] = _val;
			}
			// Sets a material reaction for ECMaterial name _materialName with coefficient _val of neighborhood order _order in {0, 1}
			// Neighborhood order equal to 0 denotes a reaction in a site; neighborhood order equal to 1 denotes a reaction with a first-order neighborhood site
			// Registers material reaction if not previously registered
			void setMaterialReactionCoefficientByName(std::string _materialName, float _val, int _order=0) {
				if (_order < 0 || _order>1) return;
				std::map<std::string, std::vector<float> >::iterator mitr = materialReactionCoefficients.find(_materialName);
				std::vector<float> thisVec(2, 0.0);
				if (mitr == materialReactionCoefficients.end()) {
					cerr << "Registering material reaction coefficient for pair (" << ECMaterialName << ", " << _materialName << "): " << _val << endl;
					thisVec[_order] = _val;
					materialReactionCoefficients.insert(make_pair(_materialName, thisVec));
				}
				else {
					thisVec = mitr->second;
					thisVec[_order] = _val;
					materialReactionCoefficients[_materialName] = thisVec;
				}
			}
			// Set a material diffusion coefficient
			void setMaterialDiffusionCoefficient(float _val) { materialDiffusionCoefficient = _val; }
			// Sets a cell proliferation response for cell type _cellType with coefficient _val
			// Registers response if not previously registered
			// _val must be greater than zero
			void setCellTypeCoefficientsProliferation(std::string _cellType, float _val) {
				if (_val < 0) return;
				std::map<std::string, float>::iterator mitr = CellTypeCoefficientsProliferation.find(_cellType);
				if (mitr == CellTypeCoefficientsProliferation.end()) {
					cerr << "Registering proliferation response coefficient for pair (" << ECMaterialName << ", " << _cellType << "): " << _val << endl;
					CellTypeCoefficientsProliferation.insert(make_pair(_cellType, _val));
				}
				else CellTypeCoefficientsProliferation[_cellType] = _val;
			}
			// Sets a cell differentiation response for cell type _cellType to cell type _newCellType with coefficient _val
			// Registers response if not previously registered
			// _val must be greater than zero
			void setCellTypeCoefficientsDifferentiation(std::string _cellType, std::string _newCellType, float _val) {
				if (_val < 0) return;
				std::map<std::string, std::map<std::string, float> >::iterator mitr = CellTypeCoefficientsDifferentiation.find(_cellType);
				std::map<std::string, float> newCoeffs;
				if (mitr == CellTypeCoefficientsDifferentiation.end()) {
					cerr << "Registering differentiation response coefficient for pair (" << ECMaterialName << ", " << _cellType << ", " << _newCellType << "): " << _val << endl;
					newCoeffs.insert(make_pair(_newCellType, _val));
					CellTypeCoefficientsDifferentiation.insert(make_pair(_cellType, newCoeffs));
				}
				else {
					std::map<std::string, float>thisCellTypeCoefficientsDifferentiation = CellTypeCoefficientsDifferentiation[_cellType];
					std::map<std::string, float>::iterator mmitr = thisCellTypeCoefficientsDifferentiation.find(_newCellType);
					if (mmitr == thisCellTypeCoefficientsDifferentiation.end()) {
						cerr << "Registering differentiation response coefficient for pair (" << ECMaterialName << ", " << _cellType << ", " << _newCellType << "): " << _val << endl;
						thisCellTypeCoefficientsDifferentiation.insert(make_pair(_newCellType, _val));
						CellTypeCoefficientsDifferentiation[_cellType] = thisCellTypeCoefficientsDifferentiation;
					}
					else CellTypeCoefficientsDifferentiation[_cellType][_newCellType] = _val;
				}
			}
			// Sets a cell death response for cell type _cellType with coefficient _val
			// Registers response if not previously registered
			// _val must be greater than zero
			void setCellTypeCoefficientsDeath(std::string _cellType, float _val) {
				if (_val < 0) return;
				std::map<std::string, float>::iterator mitr = CellTypeCoefficientsDeath.find(_cellType);
				if (mitr == CellTypeCoefficientsDeath.end()) {
					cerr << "Registering death response coefficient for pair (" << ECMaterialName << ", " << _cellType << "): " << _val << endl;
					CellTypeCoefficientsDeath.insert(make_pair(_cellType, _val));
				}
				else CellTypeCoefficientsDeath[_cellType] = _val;
			}
			// Returns a field reaction coefficient for field name _fieldName
			float getToFieldReactionCoefficientByName(std::string _fieldName) {
				std::map<std::string, float>::iterator mitr = toFieldReactionCoefficients.find(_fieldName);
				if (mitr == toFieldReactionCoefficients.end()) {
					cerr << "Warning: requested unregistered reaction coefficients for pair (" << ECMaterialName << ", " << _fieldName << ")" << endl;
					return 0.0;
				}
				else return toFieldReactionCoefficients[_fieldName];
			}
			// Returns a material reaction coefficient for field name _fieldName
			float getFromFieldReactionCoefficientByName(std::string _fieldName) {
				std::map<std::string, float>::iterator mitr = fromFieldReactionCoefficients.find(_fieldName);
				if (mitr == fromFieldReactionCoefficients.end()) {
					cerr << "Warning: requested unregistered reaction coefficients for pair (" << ECMaterialName << ", " << _fieldName << ")" << endl;
					return 0.0;
				}
				else return fromFieldReactionCoefficients[_fieldName];
			}
			// Returns a vector of material reaction coefficients for material name _material name
			// Vector is ordered by ascending neighborhood order
			std::vector<float> getMaterialReactionCoefficientsByName(std::string _materialName) {
				std::map<std::string, std::vector<float> >::iterator mitr = materialReactionCoefficients.find(_materialName);
				if (mitr == materialReactionCoefficients.end()) {
					cerr << "Warning: requested unregistered reaction coefficients for pair (" << ECMaterialName << ", " << _materialName << ")" << endl;
					return std::vector<float>(2, 0.0);
				}
				else return materialReactionCoefficients[_materialName];
			}
			// Returns a material reaction coefficient for material name _material name and neighborhood order _order
			float getMaterialReactionCoefficientByNameAndOrder(std::string _materialName, int _order) {
				if (_order < 0 || _order>1) return 0.0;
				std::vector<float> thisVec = getMaterialReactionCoefficientsByName(_materialName);
				return thisVec[_order];
			}
			float getMaterialDiffusionCoefficient() { return materialDiffusionCoefficient; }
			// Returns cell proliferation response for cell type _cellType
			float getCellTypeCoefficientsProliferation(std::string _cellType) {
				std::map<std::string, float>::iterator mitr = CellTypeCoefficientsProliferation.find(_cellType);
				if (mitr == CellTypeCoefficientsProliferation.end()) {
					cerr << "Warning: requested unregistered proliferation response coefficient for pair (" << ECMaterialName << ", " << _cellType << ")" << endl;
					return 0.0;
				}
				else return CellTypeCoefficientsProliferation[_cellType];
			}
			// Returns cell differentiation response for cell type _cellType into cell type _newCellType
			float getCellTypeCoefficientsDifferentiation(std::string _cellType, std::string _newCellType) {
				std::map<std::string, std::map<std::string, float> >::iterator mitr = CellTypeCoefficientsDifferentiation.find(_cellType);
				if (mitr == CellTypeCoefficientsDifferentiation.end()) {
					cerr << "Warning: requested unregistered differentiation response coefficient for pair (" << ECMaterialName << ", " << _cellType << ", " << _newCellType << ")" << endl;
					return 0.0;
				}
				else {
					std::map<std::string, float>::iterator mmitr = CellTypeCoefficientsDifferentiation[_cellType].find(_newCellType);
					if (mmitr == CellTypeCoefficientsDifferentiation[_cellType].end()) {
						cerr << "Warning: requested unregistered differentiation response coefficient for pair (" << ECMaterialName << ", " << _cellType << ", " << _newCellType << ")" << endl;
						return 0.0;
					}
					else return CellTypeCoefficientsDifferentiation[_cellType][_newCellType];
				}
			}
			// Returns cell death response for cell type _cellType
			float getCellTypeCoefficientsDeath(std::string _cellType) {
				std::map<std::string, float>::iterator mitr = CellTypeCoefficientsDeath.find(_cellType);
				if (mitr == CellTypeCoefficientsDeath.end()) {
					cerr << "Warning: requested unregistered death response coefficient for pair (" << ECMaterialName << ", " << _cellType << ")" << endl;
					return 0.0;
				}
				else return CellTypeCoefficientsDeath[_cellType];
			}
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
