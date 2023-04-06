#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <CompuCell3D/CompuCellLibDLLSpecifier.h>

#include "CC3DExceptions.h"
#include "PluginManager.h"
#include "Plugin.h"
#include "PluginBase.h"
#include "RandomNumberGenerators.h"
#include "Steppable.h"
#include <map>
#include <vector>
#include <string>
#include <CompuCell3D/Potts3D/Potts3D.h>

#include <CompuCell3D/ParseData.h>
#include <CompuCell3D/PottsParseData.h>
#include <CompuCell3D/ParserStorage.h>
#include <CompuCell3D/CC3DEvents.h>
//#include <QtWrappers/StreamRedirectors/CustomStreamBuffers.h>



class CC3DXMLElement;

class CustomStreamBufferBase;

namespace CompuCell3D {
    class ClassRegistry;

    class BoundaryStrategy;

    template<typename Y>
    class Field3DImpl;

    class Serializer;

    class PottsParseData;

    class ParallelUtilsOpenMP;

    class COMPUCELLLIB_EXPORT Simulator : public Steppable {

        ClassRegistry *classRegistry;

        Potts3D potts;

        int currstep;

        bool simulatorIsStepping;
        bool readPottsSectionFromXML;
        std::map<std::string, Field3D < float>*>
        concentrationFieldNameMap;
        //map of steerable objects
        std::map<std::string, SteerableObject *> steerableObjectMap;

        std::vector<Serializer *> serializerVec;
        std::string recentErrorMessage;
        bool newPlayerFlag;

        std::streambuf *cerrStreamBufOrig;
        std::streambuf *coutStreamBufOrig;
        CustomStreamBufferBase *qStreambufPtr;

        std::string basePath;
        bool restartEnabled;
        std::string step_output;
        RandomNumberGeneratorFactory rngFactory;

    public:

        ParserStorage ps;
        PottsParseData *ppdCC3DPtr;
        PottsParseData ppd;
        PottsParseData *ppdPtr;
        ParallelUtilsOpenMP *pUtils;
        // stores same information as pUtils but assumes that we use only single CPU - used in modules
        // for which user requests single CPU runs e.g. Potts with large cells
        ParallelUtilsOpenMP *pUtilsSingle;
        //making this public to avoid cross-platform compilation.linking issues.
        // todo - fix it in the next release
        std::string output_directory;

        double simValue;

        /**
         * Sets simulation output directory
         * @param output_directory [in] output directory
         */
        void setOutputDirectory(std::string output_directory);

        /**
         * returns output directory
         * @return output directory
         */
        std::string getOutputDirectory();

        /**
         * returns rng seed provided by the user or generates purely random RNG seed
         * @return rng seed - unsignewd int
         */
        virtual unsigned int getRNGSeed();

        void setOutputRedirectionTarget(ptrdiff_t _ptr);

        ptrdiff_t getCerrStreamBufOrig();

        void restoreCerrStreamBufOrig(ptrdiff_t _ptr);


        void setRestartEnabled(bool _restartEnabled) { restartEnabled = _restartEnabled; }

        bool getRestartEnabled() { return restartEnabled; }

        static PluginManager<Plugin> pluginManager;
        static PluginManager<Steppable> steppableManager;
        static PluginManager<PluginBase> pluginBaseManager;

        Simulator();

        virtual ~Simulator();

        void add_step_output(const std::string &s);

        std::string get_step_output();

        //Error handling functions
        std::string getRecentErrorMessage() { return recentErrorMessage; }

        void setNewPlayerFlag(bool _flag) { newPlayerFlag = _flag; }

        bool getNewPlayerFlag() { return newPlayerFlag; }

        std::string getBasePath() { return basePath; }

        void setBasePath(std::string _bp) { basePath = _bp; }

        ParallelUtilsOpenMP *getParallelUtils() { return pUtils; }

        ParallelUtilsOpenMP *getParallelUtilsSingleThread() { return pUtilsSingle; }

        /**
         * returns BoundaryStrategy object
         * @return BoundaryStrategy object
         */

        BoundaryStrategy *getBoundaryStrategy();

        void registerSteerableObject(SteerableObject *) ;

        void unregisterSteerableObject(const std::string &);

        SteerableObject *getSteerableObject(const std::string &_objectName);

        void setNumSteps(unsigned int _numSteps) { ppdCC3DPtr->numSteps = _numSteps; }

        unsigned int getNumSteps() { return ppdCC3DPtr->numSteps; }

        int getStep() { return currstep; }

        void setStep(int currstep) { this->currstep = currstep; }

        bool isStepping() { return simulatorIsStepping; }

        double getFlip2DimRatio() { return ppdCC3DPtr->flip2DimRatio; }

        void setRandomSeed(unsigned int seed) { ppdCC3DPtr->RandomSeed(seed); }

        unsigned int getRandomSeed() { return ppdCC3DPtr->seed; }

        // Client is responsible for deallocation.
        virtual RandomNumberGenerator *generateRandomNumberGenerator(const unsigned int &seed = 1);

        virtual RandomNumberGenerator *getRandomNumberGeneratorInstance(const unsigned int &seed = 1);

        Potts3D *getPotts() { return &potts; }

        Simulator *getSimulatorPtr() { return this; }

        ClassRegistry *getClassRegistry() { return classRegistry; }

        std::string formatErrorMessage(const CC3DException &e);

        void registerConcentrationField(std::string _name, Field3D<float> *_fieldPtr);

        std::map<std::string, Field3D < float>*> &

        getConcentrationFieldNameMap() {
            return concentrationFieldNameMap;
        }

        void postEvent(CC3DEvent &_ev);

        std::vector <std::string> getConcentrationFieldNameVector();

        Field3D<float> *getConcentrationFieldByName(std::string _fieldName);

        void registerSerializer(Serializer *_serializerPtr) { serializerVec.push_back(_serializerPtr); }

        virtual void serialize();

        // Begin Steppable interface
//        virtual void start() ;
        virtual void start();

        virtual void extraInit() ;///initialize plugins after all steppables have been initialized
		virtual void step(const unsigned int currentStep) ;

        virtual void finish() ;
        // End Steppable interface

        //these two functions are necessary to implement proper cleanup after the simulation
        //1. First it cleans cell inventory, deallocating all dynamic attributes - this has to be done before unloading modules
        //2. It unloads dynamic CC3D modules - pluginsd and steppables
        void cleanAfterSimulation();

        //unloads all the plugins - plugin destructors are called
        void unloadModules();


        void initializePottsCC3D(CC3DXMLElement *_xmlData) ;

        void processMetadataCC3D(CC3DXMLElement *_xmlData);

        void initializeCC3D() ;

        void setPottsParseData(PottsParseData *_ppdPtr) { ppdPtr = _ppdPtr; }

        CC3DXMLElement *getCC3DModuleData(std::string _moduleType, std::string _moduleName = "");

        void updateCC3DModule(CC3DXMLElement *_element);

        void steer();

    };
};
#endif
