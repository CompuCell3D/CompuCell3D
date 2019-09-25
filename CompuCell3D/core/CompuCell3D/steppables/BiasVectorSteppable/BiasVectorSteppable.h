

#ifndef BIASVECTORSTEPPABLESTEPPABLE_H

#define BIASVECTORSTEPPABLESTEPPABLE_H



#include <CompuCell3D/CC3D.h>







#include "BiasVectorSteppableDLLSpecifier.h"





namespace CompuCell3D {

    

  template <class T> class Field3D;

  template <class T> class WatchableField3D;



    class Potts3D;

    class Automaton;

    class BoundaryStrategy;

    class CellInventory;

    class CellG;

  

  class BIASVECTORSTEPPABLE_EXPORT BiasVectorSteppable : public Steppable {



                    

    WatchableField3D<CellG *> *cellFieldG;

    Simulator * sim;

    Potts3D *potts;

    CC3DXMLElement *xmlData;

    Automaton *automaton;

    BoundaryStrategy *boundaryStrategy;

    CellInventory * cellInventoryPtr;

    

    Dim3D fieldDim;

	enum StepType { STEP3D = 0, STEP2DX = 1, STEP2DY = 2, STEP2DZ = 3 };
	StepType stepType;

	enum NoiseType {VEC_GEN_WHITE3D = 0, VEC_GEN_WHITE2D = 1};
	NoiseType noiseType;

	enum BiasType {WHITE = 0, // b = white noise
				   MOMENTUM = 1, // b(t+1) = a*b(t) + (1-a)*noise
				   MANUAL = 101, // for changing b in python
				   CUSTOM = 102};// for muExpressions
	BiasType biasType;



	typedef void (BiasVectorSteppable::*step_t)(const unsigned int currentStep);
	BiasVectorSteppable::step_t stepFcnPtr;

	typedef vector<double>(BiasVectorSteppable::*noise_t)();
	BiasVectorSteppable::noise_t noiseFcnPtr;

    

  public:

    BiasVectorSteppable ();

    virtual ~BiasVectorSteppable ();

    // SimObject interface

    virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData=0);

    virtual void extraInit(Simulator *simulator);



    

    

    //steppable interface

    virtual void start();

    virtual void step(const unsigned int currentStep);

	void step_3d(const unsigned int currentStep);  // remove virtual, same for the next steps
	void step_2d_x(const unsigned int currentStep); // for x == 1
	void step_2d_y(const unsigned int currentStep); // for y == 1
	void step_2d_z(const unsigned int currentStep); // for z == 1
	void step_momentum_noise_3d(const double alpha, CellG * cell);

	void step_momentum_noise_2d_x(const double alpha, CellG * cell);

	void step_momentum_noise_2d_y(const double alpha, CellG * cell);

	void step_momentum_noise_2d_z(const double alpha, CellG * cell);


	virtual vector<double> noise_vec_generator();

	vector<double> white_noise_2d();

	vector<double> white_noise_3d();


    virtual void finish() {}





    //SteerableObject interface

    virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag=false);

    virtual std::string steerableName();

	 virtual std::string toString();



	 

  };

};

#endif        

