#ifndef rrStringListContainerH
#define rrStringListContainerH
#include <vector>
#include <string>
#include "rrObject.h"
#include "rrStringList.h"
using std::vector;
using std::string;

namespace rr
{

class RR_DECLSPEC StringListContainer : public rrObject
{
    protected:
//        string                          mLabel;
        vector<StringList>              mContainer;

    public:
                                        StringListContainer();
                                        StringListContainer(const string& lbl, const StringListContainer& cp);
                                        StringListContainer(const StringListContainer& cp);
                                        StringListContainer(const StringList& cp);
        void                            operator=(const StringListContainer& rhs);
        void                            Add(const StringListContainer& lists);
//        void                            Add(const string& lbl, const StringListContainer& lists);
        void                            Add(const string& listName, const StringList& coll);
        void                            Add(const StringList& coll);
        void                            Add(const string& coll);
        void                            Add(const int& coll);
        int                             Count() const;
        int                             ListCount() const;
        int                             TotalCount() const;
        StringList&                     operator[](const int& index);
        const StringList&               operator[](const int& index) const;
        vector<StringList>::iterator    begin();
        vector<StringList>::iterator    end();
};

ostream& operator<<(ostream& stream, const StringListContainer& list);

typedef StringListContainer ArrayList; //ArrayList is C# name..
typedef StringListContainer StringCollection;
}
#endif
