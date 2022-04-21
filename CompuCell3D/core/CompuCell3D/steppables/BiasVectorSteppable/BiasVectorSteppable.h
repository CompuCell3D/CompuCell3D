#ifndef BIASVECTORSTEPPABLESTEPPABLE_H
#define BIASVECTORSTEPPABLESTEPPABLE_H


#include <CompuCell3D/CC3D.h>


#include "BiasVectorSteppableDLLSpecifier.h"


namespace CompuCell3D {


    template<class T>
    class Field3D;

    template<class T>
    class WatchableField3D;


    class Potts3D;

    class Automaton;

    class BoundaryStrategy;

    class CellInventory;

    class CellG;

    class BIASVECTORSTEPPABLE_EXPORT BiasPersistParam {
    public:
        // add destructor, default initializer
        //BiasMomenParam() : persistentAlpha(0.0) {}
        BiasPersistParam() {
            persistentAlpha = 0;
            typeName = "";
        };
        double persistentAlpha;
        std::string typeName;

        virtual ~BiasPersistParam() {};
        /*{
            persistentAlpha = 0;
            typeName = "";
        };*/
        /* {
             delete persistentAlpha;
         };*/
    };

    // BiasMomenParam::~BiasMomenParam() {};

    /*BiasMomenParam::~BiasMomenParam()
    {
        delete persistentAlpha;
    }
        */

    class BIASVECTORSTEPPABLE_EXPORT BiasVectorSteppable : public Steppable {


        WatchableField3D<CellG *> *cellFieldG;

        Simulator *sim;

        Potts3D *potts;

        CC3DXMLElement *xmlData;

        Automaton *automaton;

        BoundaryStrategy *boundaryStrategy;

        CellInventory *cellInventoryPtr;

        std::string steppableName = "BiasVectorSteppable";

        Dim3D fieldDim;

        ParallelUtilsOpenMP *pUtils;


        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

        enum FieldType {
            FTYPE3D = 0, FTYPE2DX = 1, FTYPE2DY = 2, FTYPE2DZ = 3
        };
        FieldType fieldType;

        enum NoiseType {
            VEC_GEN_WHITE3D = 0, VEC_GEN_WHITE2D = 1
        };
        NoiseType noiseType;

        enum BiasType {
            WHITE = 0, // b = white noise
            PERSISTENT = 1, // b(t+1) = a*b(t) + (1-a)*noise
            MANUAL = 101, // for changing b in python
            CUSTOM = 102
        };// for muExpressions
        BiasType biasType;


        std::unordered_map<unsigned char, BiasPersistParam> biasPersistParamMap;


        typedef void (BiasVectorSteppable::*step_t)(const unsigned int currentStep);

        BiasVectorSteppable::step_t stepFcnPtr;

        typedef vector<double>(BiasVectorSteppable::*noise_t)();

        BiasVectorSteppable::noise_t noiseFcnPtr;

        typedef void (BiasVectorSteppable::*mom_gen_t)(const double alpha, CellG *cell);

        BiasVectorSteppable::mom_gen_t perGenFcnPtr;

        bool rnd_inited = false;

        void randomize_initial_bias();// (CellG * cell);//, bool rnd_inited);

    public:

        BiasVectorSteppable();

        virtual ~BiasVectorSteppable();

        // SimObject interface





        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);







        //steppable interface

        virtual void start();

        virtual void step(const unsigned int currentStep);

        void step_white_3d(const unsigned int currentStep);  // remove virtual, same for the next steps
        void step_white_2d_x(const unsigned int currentStep); // for x == 1
        void step_white_2d_y(const unsigned int currentStep); // for y == 1
        void step_white_2d_z(const unsigned int currentStep); // for z == 1

        void step_persistent_bias(const unsigned int currentStep); // for the momentum bias

        virtual void gen_persistent_bias(const double alpha, CellG *cell);

        void output_test(const double alpha, const CellG *cell, const vector<double> noise);

        void gen_persistent_bias_3d(const double alpha, CellG *cell);

        void gen_persistent_bias_2d_x(const double alpha, CellG *cell);

        void gen_persistent_bias_2d_y(const double alpha, CellG *cell);

        void gen_persistent_bias_2d_z(const double alpha, CellG *cell);


        virtual vector<double> noise_vec_generator();

        vector<double> white_noise_2d();

        vector<double> white_noise_3d();


        virtual void finish() {}





        //SteerableObject interface

        void determine_bias_type(CC3DXMLElement *_xmlData);

        void determine_noise_generator();

        void determine_field_type();

        void set_white_step_function();

        void set_persitent_step_function(CC3DXMLElement *_xmlData);

        void set_step_function(CC3DXMLElement *_xmlData);

        //virtual void new_update(CC3DXMLElement * _xmlData, bool _fullInitFlag);

        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();


    };

};

#endif        

