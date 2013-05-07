#ifndef rrArrayList2H
#define rrArrayList2H
#include <vector>
#include <string>
#include <list>
#include <iostream>
#include "rrArrayListItem.h"
#include "rrArrayList.h"
using namespace std;
namespace rr
{

class RR_DECLSPEC ArrayList2 : public rrObject
{
    protected:
    public:
        vector< ArrayListItemBase* >		mList; //List of ArrayListItemBase items

    public:
                                            ArrayList2();
                                            ArrayList2(const ArrayList2& cpyMe);
                                           ~ArrayList2();
        unsigned int                        Count() const;
        void                                Clear();
        void                                Add(const int& item);
        void                                Add(const double& item);
        void                                Add(const string& item);
        void                                Add(const ArrayList2& item);
        void                                Add(const ArrayListItem<ArrayList2Item>& item);

        //String lists and obsolete StringArrayLists ...
        void                                Add(const StringList& list);
        void                                Add(const string& lbl, const StringList& lists);
        void                                Add(const string& lbl, const StringArrayList& lists);

        const ArrayListItemBase&            operator[](int pos) const;
        ArrayListItemBase&                  operator[](int pos);
        void                                operator = (const ArrayList2& rhs);
        StringList                          GetSubList(const string& lName);
};


RR_DECLSPEC ostream& operator<<(ostream& stream, const ArrayList2& list);

}
#endif
