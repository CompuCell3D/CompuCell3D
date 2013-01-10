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

#ifndef SIMULATOR_H
#define SIMULATOR_H

//#include <CompuCell3D/dllDeclarationSpecifier.h>
#include <CompuCell3D/CompuCellLibDLLSpecifier.h>


#include "PluginManager.h"
#include "Plugin.h"
#include "PluginBase.h"
#include "Steppable.h"
#include <map>
#include <vector>
#include <string>
#include <CompuCell3D/Potts3D/Potts3D.h>

#include <CompuCell3D/ParseData.h>
#include <CompuCell3D/PottsParseData.h>
#include <CompuCell3D/ParserStorage.h>
//#include <QtWrappers/StreamRedirectors/CustomStreamBuffers.h>



class CC3DXMLElement;
class CustomStreamBufferBase;

namespace CompuCell3D {
	class ClassRegistry;
	class BoundaryStrategy;

	template <typename Y> class Field3DImpl;
	class Serializer;
	class PottsParseData;
	class ParallelUtilsOpenMP;

	class COMPUCELLLIB_EXPORT Simulator : public Steppable {

		ClassRegistry *classRegistry;

		Potts3D potts;

		int currstep;

		bool simulatorIsStepping;
		bool readPottsSectionFromXML;
		std::map<std::string,Field3DImpl<float>*> concentrationFieldNameMap;
		//map of steerable objects
		std::map<std::string,SteerableObject *> steerableObjectMap;

		std::vector<Serializer*> serializerVec;
		std::string recentErrorMessage;
		bool newPlayerFlag;

		std::streambuf * cerrStreamBufOrig;
		std::streambuf * coutStreamBufOrig;
		CustomStreamBufferBase * qStreambufPtr;

		std::string basePath;
		bool restartEnabled;


	public:

		ParserStorage ps;
		PottsParseData * ppdCC3DPtr;
		PottsParseData ppd;
		PottsParseData *ppdPtr;
		ParallelUtilsOpenMP *pUtils;

		double simValue;
		
		void setOutputRedirectionTarget(long  _ptr);
		long getCerrStreamBufOrig();
		void restoreCerrStreamBufOrig(long _ptr);

		void setRestartEnabled(bool _restartEnabled){restartEnabled=_restartEnabled;}
		bool getRestartEnabled(){return restartEnabled;}

		static PluginManager<Plugin> pluginManager;
		static PluginManager<Steppable> steppableManager;
		static BasicPluginManager<PluginBase> pluginBaseManager;
		
		PluginManager<Steppable> mySteppableManager;
		
		Simulator();
		virtual ~Simulator();
		//     PluginManager::plugins_t & getPluginMap(){return pluginManager.getPluginMap();}

		//Error handling functions
		std::string getRecentErrorMessage(){return recentErrorMessage;}
		void setNewPlayerFlag(bool _flag){newPlayerFlag=_flag;}
		bool getNewPlayerFlag(){return newPlayerFlag;}

		std::string getBasePath(){return basePath;}
		void setBasePath(std::string _bp){basePath=_bp;}

		ParallelUtilsOpenMP * getParallelUtils(){return pUtils;}


		BoundaryStrategy * getBoundaryStrategy();
		void registerSteerableObject(SteerableObject *);
		void unregisterSteerableObject(const std::string & );
		SteerableObject * getSteerableObject(const std::string & _objectName);

		unsigned int getNumSteps() {return ppdCC3DPtr->numSteps;}
		int getStep() {return currstep;}
		bool isStepping(){return simulatorIsStepping;}
		double getFlip2DimRatio(){return ppdCC3DPtr->flip2DimRatio;}
		Potts3D *getPotts() {return &potts;}
		Simulator *getSimulatorPtr(){return this;}
		ClassRegistry *getClassRegistry() {return classRegistry;}


		void registerConcentrationField(std::string _name,Field3DImpl<float>* _fieldPtr);
		std::map<std::string,Field3DImpl<float>*> & getConcentrationFieldNameMap(){
			return concentrationFieldNameMap;
		}
		std::vector<std::string> getConcentrationFieldNameVector();
		Field3DImpl<float>* getConcentrationFieldByName(std::string _fieldName);
		
		void registerSerializer(Serializer * _serializerPtr){serializerVec.push_back(_serializerPtr);}
		virtual void serialize();

		// Begin Steppable interface
		virtual void start();
		virtual void extraInit();///initialize plugins after all steppables have been initialized
		virtual void step(const unsigned int currentStep);
		virtual void finish();
		// End Steppable interface

		void unloadModules();
		void initializePottsCC3D(CC3DXMLElement * _xmlData);
		void processMetadataCC3D(CC3DXMLElement * _xmlData);

		void initializeCC3D();
		void setPottsParseData(PottsParseData * _ppdPtr){ppdPtr=_ppdPtr;}
		CC3DXMLElement * getCC3DModuleData(std::string _moduleType,std::string _moduleName="");
		void updateCC3DModule(CC3DXMLElement *_element);
		void steer();
		
	};
};
#endif
