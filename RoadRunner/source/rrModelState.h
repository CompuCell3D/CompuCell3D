#ifndef rrModelStateH
#define rrModelStateH
#include <ostream>
#include <istream>
#include <vector>
#include "rrObject.h"
#include "rrModelFromC.h"

using std::vector;
using std::istream;
using std::ostream;

namespace rr
{
class RR_DECLSPEC ModelState : public rrObject
{
    protected:
    public:
        //Todo:  Lot of stuff is wrong here.. ??
        double                          mTime;
        vector<double>                  mBoundarySpeciesConcentrations;
        vector<double>                  mCompartmentVolumes;
        vector<double>                  mConservedTotals;
        vector<double>                  mDyDt;
        vector<double>                  mFloatingSpeciesConcentrations;
        vector<double>                  mGlobalParameters;
        vector<bool>                    mPreviousEventStatusArray;
        vector<double>                  mRateRules;
        vector<double>                  mRates;
        vector<double>                  mModifiableSpeciesReferences;
        vector<bool>                    mEventStatusArray;
        vector<double>                  mEventTests;

        void                            InitializeFromModel(ModelFromC& model);

    public:
                                        ModelState(ModelFromC& model);
        void                            WriteTo(const string& fileName);
        static ModelState               ReadFrom(istream& stream);
        static ModelState               ReadFrom(const string& fileName);
        void                            WriteTo(ostream& stream);
        void                            AssignToModel(ModelFromC& model);
};

}
#endif

//////////////////////////////////////////////////////////////////
//c#
//using System;
//using System.IO;
//using System.Runtime.Serialization.Formatters.Binary;
//
//namespace LibRoadRunner
//{
//    [Serializable]
//    public class ModelState
//    {
//        private double[] _BoundarySpeciesConcentrations;
//        private double[] _CompartmentVolumes;
//        private double[] _ConservedTotals;
//        private double[] _DyDt;
//        private bool[] _EventStatusArray;
//        private double[] _EventTests;
//        private double[] _FloatingSpeciesConcentrations;
//        private double[] _GlobalParameters;
//        private bool[] _PreviousEventStatusArray;
//        private double[] _RateRules;
//        private double[] _Rates;
//        private double[] _ModifiableSpeciesReferences;
//        private double _Time;
//
//        public ModelState(IModel model)
//        {
//            InitializeFromModel(model);
//        }
//
//        public bool[] EventStatusArray
//        {
//            get
//            {
//                return _EventStatusArray;
//            }
//            set
//            {
//                _EventStatusArray = value;
//            }
//        }
//
//        public double[] ModifiableSpeciesReferences
//        {
//            get { return _ModifiableSpeciesReferences; }
//            set { _ModifiableSpeciesReferences = value; }
//        }
//
//        public double[] FloatingSpeciesConcentrations
//        {
//            get { return _FloatingSpeciesConcentrations; }
//            set { _FloatingSpeciesConcentrations = value; }
//        }
//
//        public double[] BoundarySpeciesConcentrations
//        {
//            get { return _BoundarySpeciesConcentrations; }
//            set { _BoundarySpeciesConcentrations = value; }
//        }
//
//        public double[] CompartmentVolumes
//        {
//            get { return _CompartmentVolumes; }
//            set { _CompartmentVolumes = value; }
//        }
//
//        public double[] GlobalParameters
//        {
//            get { return _GlobalParameters; }
//            set { _GlobalParameters = value; }
//        }
//
//        public double[] ConservedTotals
//        {
//            get { return _ConservedTotals; }
//            set { _ConservedTotals = value; }
//        }
//
//        public double[] DyDt
//        {
//            get { return _DyDt; }
//            set { _DyDt = value; }
//        }
//
//        public double[] Rates
//        {
//            get { return _Rates; }
//            set { _Rates = value; }
//        }
//
//        public double[] RateRules
//        {
//            get { return _RateRules; }
//            set { _RateRules = value; }
//        }
//
//        public double Time
//        {
//            get { return _Time; }
//            set { _Time = value; }
//        }
//
//        public void WriteTo(string fileName)
//        {
//            var stream = new FileStream(fileName, FileMode.Create);
//            WriteTo(stream);
//        }
//
//        public static ModelState ReadFrom(Stream stream)
//        {
//            var formatter = new BinaryFormatter();
//            var state = (ModelState) formatter.Deserialize(stream);
//            stream.Close();
//            return state;
//        }
//
//        public static ModelState ReadFrom(string fileName)
//        {
//            var stream = new FileStream(fileName, FileMode.Open);
//            return ReadFrom(stream);
//        }
//
//        public void WriteTo(Stream stream)
//        {
//            var formatter = new BinaryFormatter();
//            formatter.Serialize(stream, this);
//            stream.Flush();
//            stream.Close();
//        }
//
//        public void AssignToModel(IModel model)
//        {
//            model.y = _FloatingSpeciesConcentrations;
//            model.bc = _BoundarySpeciesConcentrations;
//            model.c = _CompartmentVolumes;
//            model.gp = _GlobalParameters;
//            model.ct = _ConservedTotals;
//
//            model.dydt = _DyDt;
//            model.rates = _Rates;
//            model.rateRules = _RateRules;
//
//            model.eventTests = _EventTests;
//            model.eventStatusArray = _EventStatusArray;
//            model.previousEventStatusArray = _PreviousEventStatusArray;
//            model.time = _Time;
//            model.convertToAmounts();
//
//            model.sr = _ModifiableSpeciesReferences;
//        }
//
//        private void InitializeFromModel(IModel model)
//        {
//            model.convertToConcentrations();
//            _FloatingSpeciesConcentrations = GetCopy(model.y);
//            _BoundarySpeciesConcentrations = GetCopy(model.bc);
//            _CompartmentVolumes = GetCopy(model.c);
//            _GlobalParameters = GetCopy(model.gp);
//            _ConservedTotals = GetCopy(model.ct);
//            _DyDt = GetCopy(model.dydt);
//            _Rates = GetCopy(model.rates);
//            _RateRules = GetCopy(model.rateRules);
//            _ModifiableSpeciesReferences = GetCopy(model.sr);
//            _Time = model.time;
//
//            _EventStatusArray = GetCopy(model.eventStatusArray);
//            _EventTests = GetCopy(model.eventTests);
//            _PreviousEventStatusArray = GetCopy(model.previousEventStatusArray);
//        }
//
//        public static double[] GetCopy(double[] oVector)
//        {
//            var oResult = new double[oVector.Length];
//            oVector.CopyTo(oResult, 0);
//            return oResult;
//        }
//
//        public static bool[] GetCopy(bool[] oVector)
//        {
//            var oResult = new bool[oVector.Length];
//            oVector.CopyTo(oResult, 0);
//            return oResult;
//        }
//    }
//}
