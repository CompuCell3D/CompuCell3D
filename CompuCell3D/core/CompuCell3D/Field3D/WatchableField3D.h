
#ifndef WATCHABLEFIELD3D_H
#define WATCHABLEFIELD3D_H

#include <vector>

#include "Field3DImpl.h"
#include "Field3DChangeWatcher.h"

#include <CompuCell3D/CC3DExceptions.h>

namespace CompuCell3D {

    template<class T>
    class Field3DImpl;

    template<class T>
    class WatchableField3D : public Field3DImpl<T> {
        std::vector<Field3DChangeWatcher<T> *> changeWatchers;

    public:
        /**
         * @param dim The field dimensions
         * @param initialValue The initial value of all data elements in the field.
         */
        WatchableField3D(const Dim3D dim, const T &initialValue) :
                Field3DImpl<T>(dim, initialValue) {}

        virtual ~WatchableField3D() {}

        virtual void addChangeWatcher(Field3DChangeWatcher<T> *watcher) {
            if (!watcher) throw CC3DException("addChangeWatcher() watcher cannot be NULL!");
            changeWatchers.push_back(watcher);
        }

        virtual void set(const Point3D &pt, const T value) {
            T oldValue = Field3DImpl<T>::get(pt);
            Field3DImpl<T>::set(pt, value);

            for (unsigned int i = 0; i < changeWatchers.size(); i++)
                changeWatchers[i]->field3DChange(pt, value, oldValue);
        }

        virtual void set(const Point3D &pt, const Point3D &addPt, const T value) {
            T oldValue = Field3DImpl<T>::get(pt);
            Field3DImpl<T>::set(pt, value);

            for (unsigned int i = 0; i < changeWatchers.size(); i++) {
                changeWatchers[i]->field3DChange(pt, value, oldValue);
                changeWatchers[i]->field3DChange(pt, addPt, value, oldValue);
            }
        }
    };
};
#endif
