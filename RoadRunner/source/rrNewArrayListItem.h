#ifndef rrNewArrayListItemH
#define rrNewArrayListItemH
#include "rrObject.h"
#include "rrStringList.h"
#include "rrNewArrayListItemObject.h"

//This unit contains
// 1) a base class for ArrayListItems
// 2) Template for basic type ArrayListItems, such as int, char double etc.
// 3) An ArrayListItem class, that represent an ArrayList, within an ArrayList object

namespace rr
{

template <class T>
class NewArrayListItem : public NewArrayListItemObject
{
    private:
        T                           mItemValue;

    public:
                                    NewArrayListItem(const T& val);

        virtual                    ~NewArrayListItem(){}
                                    operator T(){return mItemValue;}
        virtual const char          operator[](const int& pos) const {return '\0';}     //Make sense for string types
        NewArrayListItem<T>&        operator=(const NewArrayListItem<T>& rhs);
};

template<class T>
NewArrayListItem<T>::NewArrayListItem(const T& val)
:
mItemValue(val)
{}

template<class T>
NewArrayListItem<T>& NewArrayListItem<T>::operator=(const NewArrayListItem<T>& rhs)
{
    if(this != &rhs)
    {
        mItemValue = rhs.mItemValue;
    }

    return *this;
}

ostream& operator<<(ostream& stream, const NewArrayListItemObject& item);

}
#endif
