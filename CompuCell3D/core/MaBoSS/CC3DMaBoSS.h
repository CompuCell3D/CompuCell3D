/*
MaBoSS Integration
Written by T.J. Sego, Ph.D.
Biocomplexity Institute
Indiana University
Bloomington, IN, U.S.A. 
*/
#ifndef CC3DMABOSS_H
#define CC3DMABOSS_H

#include <algorithm>
#include <tuple>
#include <vector>

#include "engine/src/maboss-config.h"
#include "BooleanNetwork.h"
#include "RandomGenerator.h"
#include "RunConfig.h"
#include "StochasticSimulationEngine.h"

#include "CC3DMaBoSSDLLSpecifier.h"

namespace MaBoSSCC3D {

    class CC3DMABOSS_EXPORT CC3DRandomGenerator {

        RandomGenerator* randGen;
        int seed;

    public:

        CC3DRandomGenerator();
        CC3DRandomGenerator(const int& _seed);
        CC3DRandomGenerator(const CC3DRandomGenerator& other);
        ~CC3DRandomGenerator();

        std::string getName() const;
        bool isPseudoRandom() const;

        unsigned int generateUInt32();
        double generate();

        void setSeed(const int &_seed);
        int getSeed() const;

        RandomGenerator* getRandomGenerator();

    };


    class CC3DMABOSS_EXPORT CC3DRunConfig {

        RunConfig* runConfig;
        CC3DRandomGenerator* randGen;

    public:

        CC3DRunConfig();
        CC3DRunConfig(const CC3DRunConfig& other);
        ~CC3DRunConfig();

        RunConfig* getRunConfig();
        CC3DRandomGenerator* getRandomGenerator();

        double getTimeTick() const;
        void setTimeTick(const double &time_tick);
        unsigned int getSampleCount() const;
        void setSampleCount(const unsigned int &sample_count);
        const bool getDiscreteTime() const;
        void setDiscreteTime(const bool& discrete_time);

        void setSeed(const int &seed);
        int getSeed() const;
        unsigned int generateUInt32();
        double generate();

        int parse(Network* network, const char* file = NULL);
        int parseExpression(Network* network, const char* expr);

    };

    class CC3DMABOSS_EXPORT CC3DMaBoSSNodeAttributeAccessorPy {
        Node* node;
        const std::string attr_name;

    public:

        CC3DMaBoSSNodeAttributeAccessorPy(Node* _node, const std::string& _attr_name) : node(_node), attr_name(_attr_name) {};
        ~CC3DMaBoSSNodeAttributeAccessorPy() {};

        // Underlying accessors for properties "attr_expression" and "attr_string"

        const Expression* getExpression();
        void setExpression(const Expression* expr);
        std::string getString();
        void setString(const std::string& str);

        // SWIG support
        
        // Python support
        #ifdef SWIGPYTHON
        std::string __str__() { return getString(); }
        #endif // SWIGPYTHON

    };

    class CC3DMABOSS_EXPORT CC3DMaBoSSNode {
        Node* node;
        Network* network;
        NetworkState* networkState;
        NetworkState* networkStateInit;
        RandomGenerator* randGen;

    public:
        CC3DMaBoSSNode(Node* _node, Network* _network, NetworkState* _networkState, NetworkState* _networkStateInit, RandomGenerator* _randGen) : 
            node(_node), 
            network(_network), 
            networkState(_networkState), 
            networkStateInit(_networkStateInit), 
            randGen(_randGen)
        {}
        ~CC3DMaBoSSNode() { };

        const std::string& getLabel() const { return node->getLabel(); }
        void setDescription(const std::string& description) { node->setDescription(description); }
        const std::string& getDescription() const { return node->getDescription(); }
        void setLogicalInputExpression(const Expression* logicalInputExpr) { node->setLogicalInputExpression(logicalInputExpr); }
        void setRateUpExpression(const Expression* expr) { node->setRateUpExpression(expr); }
        void setRateDownExpression(const Expression* expr) {node->setRateDownExpression(expr); }
        const Expression* getLogicalInputExpression() const { return node->getLogicalInputExpression(); }
        const Expression* getRateUpExpression() const { return node->getRateUpExpression(); }
        const Expression* getRateDownExpression() const { return node->getRateDownExpression(); }
        void setAttributeExpression(const std::string& attr_name, const Expression* expr) { node->setAttributeExpression(attr_name, expr); }
        NodeState getIState() { return node->getNodeState(*networkStateInit);}  // node->getIState when initial state correctly tracks from config file
        void setIState(NodeState istate) { node->setNodeState(*networkStateInit, istate); }  // node->setIState when initial state correctly tracks from config file
        // bool istateSetRandomly() const { return node->istateSetRandomly(); }  // enable when initial state correctly tracks from config file
        bool isInternal() const { return node->isInternal(); }
        void setInternal(bool is_internal) { node->isInternal(is_internal); }
        bool isReference() const { return node->isReference(); }
        void setReference(bool is_reference) { node->setReference(is_reference); }
        NodeState getReferenceState() const { return node->getReferenceState(); }
        void setReferenceState(NodeState referenceState) { node->setReferenceState(referenceState); }
        const Expression* getAttributeExpression(const std::string& attr_name) const { return node->getAttributeExpression(attr_name); }
        void setAttributeString(const std::string& attr_name, const std::string& str) { node->setAttributeString(attr_name, str); }
        std::string getAttributeString(const std::string& attr_name) const { return node->getAttributeString(attr_name); }
        bool isInputNode() const { return node->isInputNode(); }
        double getRateUp() const { return node->getRateUp(*networkState); }
        double getRateDown() const { return node->getRateDown(*networkState); }
        NodeState getNodeState() const { return node->getNodeState(*networkState); }
        void setNodeState(NodeState state) { return node->setNodeState(*networkState, state); }
        bool computeNodeState(NodeState& state) const { return node->computeNodeState(*networkState, state); }
        std::string toString() const { return node->toString(); }

        // SWIG support
        
        // Python support
        #ifdef SWIGPYTHON
        // Attributes are accessible like a dictionary
        CC3DMaBoSSNodeAttributeAccessorPy __getitem__(const std::string& key) { return CC3DMaBoSSNodeAttributeAccessorPy(node, key); }

        std::string __str__() { return toString(); }
        #endif // SWIGPYTHON

    };

    class CC3DMABOSS_EXPORT CC3DMaBoSSEngine {

        double time;
        double stepSize;
        Network* network;
        CC3DRunConfig* runConfig;
        StochasticSimulationEngine* engine;

        NetworkState networkState;
        NetworkState networkStateInit;
        
        NodeIndex getTargetNode(CC3DRandomGenerator* random_generator, const MAP<NodeIndex, double>& nodeTransitionRates, double total_rate) const;

    public:

        CC3DMaBoSSEngine(const char* ctbndl_file, const char* cfg_file, const double& stepSize=1.0, const int& seed=0, const bool& cfgSeed=true);

        ~CC3DMaBoSSEngine();

        // simulate network for an amount of time
        void step(const double& _stepSize = -1.);
        // load a network
        void loadNetworkState(const NetworkState& _networkState);

        const double getTime() const { return time; }
        void setTime(const double& _time) { time = _time; }
        const double getStepSize() const { return stepSize; }
        void setStepSize(const double& _stepSize) { stepSize = _stepSize; }

        // Component accessors

        // Network
        Network* getNetwork();
        // Run configuration
        CC3DRunConfig* getRunConfig();
        // NetworkState
        NetworkState* getNetworkState();

        // Node accessor
        CC3DMaBoSSNode getNode(const std::string& label);

        // SWIG support
        
        // Python support
        #ifdef SWIGPYTHON
        CC3DMaBoSSNode __getitem__(const std::string &key) { return getNode(key); }
        void __setitem__(const std::string &key, NodeState value) { network->getNode(key)->setNodeState(networkState, value); }
        #endif // SWIGPYTHON
        

    };

};

#endif
