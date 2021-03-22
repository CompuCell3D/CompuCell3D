%module MaBoSSCC3DPy

%{
    #define SWIG_FILE_WITH_INIT

    #include "CC3DMaBoSS.h"

    using namespace MaBoSSCC3D;

%}

// C++ std::string handling
%include "std_string.i"

// C++ std::unordered_map handling
%include "std_unordered_map.i"

// C++ std::vector handling
%include "std_vector.i"

// Exception handling
%include "exception.i"

%exception {
    try {
        $action
    } catch(BNException& e) {
        SWIG_exception(SWIG_RuntimeError, e.getMessage().c_str());
    }
}

#define CC3DMABOSS_EXPORT

// Fix LogicalExprGenContext::LevelManager (BooleanNetwork.h): no nested structs
%inline %{

class LevelManager {

    LogicalExprGenContext& genctx;
    unsigned int level;

public:
    LevelManager(LogicalExprGenContext& genctx) : genctx(genctx) {
        level = genctx.getLevel();
        genctx.incrLevel();
    }

    unsigned int getLevel() const {return level;}

    ~LevelManager() {
        genctx.decrLevel();
    }
};

%}
%ignore LogicalExprGenContext::LevelManager;
%extend LogicalExprGenContext {
    %pythoncode %{
        @staticmethod
        def LevelManager(genctx):
            return LevelManager(genctx)
    %}
}

// Fix IStateGroup::ProbaIState (BooleanNetwork.h): no nested structs
%inline %{

    struct ProbaIState {
        double proba_value;
        std::vector<double>* state_value_list;

        ProbaIState(Expression* proba_expr, std::vector<Expression*>* state_expr_list) {
            NetworkState network_state;
            proba_value = proba_expr->eval(NULL, network_state);
            std::vector<Expression*>::iterator begin = state_expr_list->begin();
            std::vector<Expression*>::iterator end = state_expr_list->end();
            state_value_list = new std::vector<double>();
            while (begin != end) {
        NetworkState network_state;
        state_value_list->push_back((*begin)->eval(NULL, network_state));
        ++begin;
            }
        }

        // only one node
        ProbaIState(double proba_value, Expression* state_expr) {
            this->proba_value = proba_value;
            state_value_list = new std::vector<double>();
            NetworkState network_state;
            state_value_list->push_back(state_expr->eval(NULL, network_state));
        }

        ProbaIState(double proba_value, double istate_value) {
            this->proba_value = proba_value;
            state_value_list = new std::vector<double>();
            state_value_list->push_back(istate_value);
        }
        double getProbaValue() {return proba_value;}
        std::vector<double>* getStateValueList() {return state_value_list;}
        void normalizeProbaValue(double proba_sum) {proba_value /= proba_sum;}
    };

%}
%ignore IStateGroup::ProbaIState;
%template(Vector_Node_Ptr) std::vector<Node*>;
%template(Vector_Expression_Ptr) std::vector<Expression*>;
%extend IStateGroup {
    %pythoncode %{
        @staticmethod
        def ProbaIState(proba_expr: Expression, state_expr_list: Vector_Expression_Ptr):
            return ProbaIState(proba_expr, state_expr_list)
    %}
}

// Ignoring problematic/unsupported features
%ignore Network::operator=;
%ignore NetworkState::operator<;
%ignore operator<<;
%ignore RCin;
%ignore RCparse;
%ignore runconfig_setNetwork;
%ignore RC_set_file;
%ignore RC_set_expr;

// Generics
%pythoncode %{
    def readonly_property_setter(prop_name: str):
        def inner(self, val):
            raise AttributeError(f'Assignment of {prop_name} is illegal')
        return inner
%}

%define PROPERTYEXTENSORPY(className, propName, propGet, propSet)
%extend className {
    %pythoncode %{
        __swig_getmethods__["propName"] = propGet
        __swig_setmethods__["propName"] = propSet
        if _newclass: propName = property(propGet, propSet)
    %}
}
%enddef

%define READONLYPROPERTYEXTENSORPY(className, propName, propGet)
%extend className {
    %pythoncode %{
        __swig_getmethods__["propName"] = propGet
        __swig_setmethods__["propName"] = readonly_property_setter("propName")
        if _newclass: propName = property(propGet, readonly_property_setter("propName"))
    %}
}
%enddef

%define MABOSSTOSTRINGPY(className)
%extend className {
    %pythoncode %{
        def __str__(self):
            return self.toString()
    %}
}
%enddef

%include "BooleanNetwork.h"
%include "RandomGenerator.h"
%include "RunConfig.h"
%include "CC3DMaBoSS.h"

// Extending Exression
MABOSSTOSTRINGPY(Expression)

// Extending Node
MABOSSTOSTRINGPY(Node)

// Extending SymbolTable
READONLYPROPERTYEXTENSORPY(SymbolTable, names, getSymbolsNames)
%extend SymbolTable {
    %pythoncode %{
        def __getitem__(self, item: str):
            if not item.startswith('$'):
                item = '$' + item
            symbol = self.getSymbol(item)
            if symbol is None:
                raise KeyError(f"symbol {item} is not defined")
            return self.getSymbolValue(symbol, check=False)

        def __setitem__(self, item: str, value: float):
            if not item.startswith('$'):
                item = '$' + item
            symbol = self.getSymbol(item)
            if symbol is None:
                raise KeyError(f"symbol {item} is not defined")
            self.setSymbolValue(symbol, value)
            self.unsetSymbolExpressions()
    %}
}

// Extending Network
READONLYPROPERTYEXTENSORPY(Network, nodes, getNodes)
READONLYPROPERTYEXTENSORPY(Network, symbol_table, getSymbolTable)
MABOSSTOSTRINGPY(Network)

// Extending CC3DMaBoSSNodeAttributeAccessorPy
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNodeAttributeAccessorPy, attr_expression, getExpression, setExpression)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNodeAttributeAccessorPy, attr_string, getString, setString)

// Extending CC3DMaBoSSNode
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNode, description, getDescription, setDescription)
READONLYPROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNode, is_input, isInputNode)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNode, is_internal, isInternal, setInternal)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNode, is_reference, isReference, setReference)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNode, istate, getIState, setIState)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNode, logical_input_expr, getLogicalInputExpression, setLogicalInputExpression)
READONLYPROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNode, rate_down, getRateDown)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNode, rate_down_expr, getRateDownExpression, setRateDownExpression)
READONLYPROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNode, rate_up, getRateUp)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNode, rate_up_expr, getRateUpExpression, setRateUpExpression)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNode, ref_state, getReferenceState, setReferenceState)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSNode, state, getNodeState, setNodeState)
MABOSSTOSTRINGPY(MaBoSSCC3D::CC3DMaBoSSNode)

// Extending CC3DRunConfig
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DRunConfig, sample_count, getSampleCount, setSampleCount)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DRunConfig, seed, getSeed, setSeed)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DRunConfig, time_tick, getTimeTick, setTimeTick)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DRunConfig, discrete_time, getDiscreteTime, setDiscreteTime)

// Extending CC3DMaBoSSEngine
READONLYPROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSEngine, run_config, getRunConfig)
READONLYPROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSEngine, network, getNetwork)
READONLYPROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSEngine, time, getTime)
PROPERTYEXTENSORPY(MaBoSSCC3D::CC3DMaBoSSEngine, step_size, getStepSize, setStepSize)
