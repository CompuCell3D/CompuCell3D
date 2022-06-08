#ifndef FIELD3DCHANGEWATCHER_H
#define FIELD3DCHANGEWATCHER_H

#include "Point3D.h"


namespace CompuCell3D {

    template<class T>
    class Field3DChangeWatcher {
    public:
        /**
         * Notify the watcher that a set is being performed.
         * This function will be called after the change has occured!
         *
         * @param pt The location.
         * @param newValue The current value.
         * @param oldValue The previous value.
         */
        virtual void field3DChange(const Point3D &pt, T newValue,
                                   T oldValue) = 0;

        /**
        * Notify the watcher that a set is being performed with respect to two points.
        * This function will be called after the change has occured!
        *
        * @param pt The location of the set.
        * @param addPt Another point with respect to the set.
        * @param newValue The current value.
        * @param oldValue The previous value.
        */
        virtual void field3DChange(const Point3D &pt, const Point3D &addPt, T newValue, T oldValue) {};
    };
};
#endif
