#ifndef COMPUCELL3DCELLVELOCITYDATA_H
#define COMPUCELL3DCELLVELOCITYDATA_H

#include <CompuCell3D/Field3D/Point3D.h>
#include <Utils/cldeque.h>
#include <Utils/Coordinates3D.h>
#include <BasicUtils/BasicException.h>

#include <iostream>

namespace CompuCell3D {

/**
@author m
*/

    class CellVelocityData {
    private:

        cldeque<Coordinates3D < float> >
        ::size_type cldequeCapacity;
        cldeque<Coordinates3D < float> >
        ::size_type enoughDataThreshold;

    public:

        CellVelocityData(cldeque<Coordinates3D < float>

        >
        ::size_type _cldequeCapacity,
                cldeque<Coordinates3D < float>
        >
        ::size_type _enoughDataThreshold
        ):enoughData(false),

        cldequeCapacity (_cldequeCapacity),
        enoughDataThreshold(_enoughDataThreshold) {

            cellCOMPtr = new cldeque<Coordinates3D < float> > ();
            cellCOMPtr->assign(cldequeCapacity, Coordinates3D<float>(0., 0., 0.));
            velocity = Coordinates3D<float>(0., 0., 0.);
            timePtr = new cldeque<float>();
            timePtr->assign(cldequeCapacity, 1);

            using namespace std;
        }


        void setCldequeCapacity(cldeque<Coordinates3D < float>

        >
        ::size_type _capacity
        )
        {
            ASSERT_OR_THROW("capacity must be at least 1 ", _capacity >= 1);
            ASSERT_OR_THROW("capacity must be >= enoughDataThreshold ", _capacity >= enoughDataThreshold);
            cldequeCapacity = _capacity;

        }

        void setEnoughDataThreshold(cldeque<Coordinates3D < float>

        >
        ::size_type _threshold
        )
        {
            ASSERT_OR_THROW("EnoughDataThreshold > cldequeCapacity. Threshold too big",
                            cldequeCapacity >= _threshold);
            enoughDataThreshold = _threshold;

        }


        CellVelocityData();

        ~CellVelocityData();

        //access
        Coordinates3D<float> operator[](cldeque<Coordinates3D < float>

        >
        ::size_type idx
        ){
            return (*cellCOMPtr)[idx];
        }

        Coordinates3D<float> getVelocityData(cldeque<Coordinates3D < float>

        >
        ::size_type idx
        ){
            return (*cellCOMPtr)[idx];
        }

        void setInstantenousVelocity(Coordinates3D<float> &_vel) {
            velocity = _vel;
        }

        void setInstantenousVelocity(float x, float y, float z) {
            velocity.XRef() = x;
            velocity.YRef() = y;
            velocity.ZRef() = z;
        }

        Coordinates3D<float> getInstantenousVelocity() {
            return velocity;
        }

        void setAverageVelocity(Coordinates3D<float> &_vel) {
            velocity = _vel;
        }

        void setAverageVelocity(float x, float y, float z) {
            velocity.XRef() = x;
            velocity.YRef() = y;
            velocity.ZRef() = z;
        }

        Coordinates3D<float> getAverageVelocity() {
            return velocity;
        }


        Coordinates3D<float> getLastCM() {
            return (*cellCOMPtr)[0];
        }


        Coordinates3D<float> getLatestAverageVelocity() {
            if (enoughData) {
                return (*cellCOMPtr)[0] - (*cellCOMPtr)[1];
            } else {
                return Coordinates3D<float>(0., 0., 0.);
            }
        }

        void push_front(Coordinates3D<float> &_com) {

            cellCOMPtr->push_front(_com);

            if (enoughData) return;
            else {
                ++numberOfSamples;
                if (numberOfSamples >= enoughDataThreshold) {
                    enoughData = true;
                }
            }

        }

        void push_front(float _x, float _y, float _z) {
            Coordinates3D<float> coordinates3D(_x, _y, _z);

            cellCOMPtr->push_front(coordinates3D);

            if (enoughData) return;
            else {
                ++numberOfSamples;
                if (numberOfSamples >= enoughDataThreshold) {
                    enoughData = true;
                }
            }

        }

        void resize(cldeque<Coordinates3D < float>

        >
        ::size_type _new_size
        ){
            cellCOMPtr->setSize(_new_size);
        }

        void produceVelocityHistoryFromSource(const CellVelocityData *source);

        cldeque<Coordinates3D < float> >

        ::size_type size() {
            using namespace std;
            #include <Logger/CC3DLogger.h>
      CC3D_Log(LOG_DEBUG) << "cellCOMPtr="<<cellCOMPtr;
            return cellCOMPtr->size();
        }

        ///main member
        cldeque <Coordinates3D<float>> *cellCOMPtr;

        cldeque<float> *timePtr;

        Coordinates3D<float> velocity;

        bool enoughData;
        unsigned short numberOfSamples;

    private:


    };


};

#endif
