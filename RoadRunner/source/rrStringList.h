#ifndef rrStringListH
#define rrStringListH
#include <vector>
#include <string>
#include "rrObject.h"

using std::vector;
using std::string;
using std::ostream;

namespace rr
{

class RR_DECLSPEC StringList : public rrObject
{
    protected:

        vector<string>              mStrings;
        vector<string>::iterator    mLI;

    public:
                                    StringList();
                                    StringList(const string& str, const string& delimiters = ", ");
                                    StringList(const vector<string>& strings);
                                    StringList(const StringList& cp);
                                   ~StringList();

        void                        InsertAt(const int& index, const string& item);
        void                        Add(const string& str);
		void 						Append(const StringList& list);
        string                      AsString(const string& delimiter = gComma) const;
        unsigned int                Count() const;
        StringList&                 operator=(const StringList& rhs);
        string&                     operator[](const int& index);
        const string&               operator[](const int& index) const;
        StringList                  operator-(const StringList& rhs);

        int                         find(const string& item);
        int                         IndexOf(const string& item);
        void                        clear();
        void                        empty();
        bool                        Contains(const string& item) const;
        bool                        DontContain(const string& item) const;
        void                        push_back(const string& item);
        vector<string>::iterator    begin();
        vector<string>::iterator    end();
        void                        PreFix(const string& fix);
        void                        PostFix(const string& fix);
		RR_DECLSPEC friend ostream& operator<<(ostream& stream, const StringList& list);
};



}
#endif
