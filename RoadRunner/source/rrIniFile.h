#ifndef IniFileH
#define IniFileH
#include <vector>
#include <fstream>
#include <string>
#include "rrObject.h"
#include "rrStringUtils.h"
#include "rrIniSection.h"
#include "rrIniKey.h"
#include "rrFileName.h"

using namespace rr;


using std::string;
using std::fstream;
using std::ios;
using std::ios_base;

namespace rr
{
const int MAX_LINE_BUFFER_SIZE = 2048;
class rrIniKey;
class rrIniSection;

// IniFile typedefs
typedef std::vector<rrIniKey*>     		KeyList;
typedef std::vector<string> 	    	NonKeyList;
typedef std::vector<rrIniSection*>  	SectionList;
typedef SectionList::iterator       	SectionItor;

class RR_DECLSPEC IniFile : public rrObject
{
	private:
		// When set, this define will cause SetValue() to create a new section, if
		// the requested section does not allready exist.
		const int			mAutoCreateSections;

		// When set, this define causes SetValue() to create a new key, if the
		// requested key does not allready exist.
		const int 			mAutoCreateKeys;
		
		string 				mCommentIndicators;
		const string 		mEqualIndicator;
		const string 		mWhiteSpace;

		// General Purpose Utility Functions
		string	            GetNextWord(string& CommandLine);
		string	            Trim(string& Str);
		int		            WriteLine(fstream& stream, const char* fmt, ...);

	protected:
		SectionList	        mSections;		// List of sections
		FileName	        mIniFileName;	// The filename to write to
		bool	            mIsDirty;		// Tracks whether or not data has changed.
        bool                mWasFound;
    	bool 		        mAutoSave;		//Save ini file automatically on destruction


	public:
		int			        mFlags;		// Our settings flags.
					        IniFile(const string& fName = gEmptyString, bool autoLoad = false, bool autoSave = false);
		virtual		        ~IniFile();

        int					GetNumberOfSections(){return mSections.size();}
        rrIniSection*		GetSection(int i){return mSections[i];}

		// File handling methods
		string              GetFilePath(){return mIniFileName.GetPath();}
		string              GetFileName(){return mIniFileName;}
        string              GetFullFileName(){return mIniFileName.GetPathAndFileName();}
        bool                SetFilePath(const string& path);
		bool		        Load(const string& fName = rr::gEmptyString);
		rrIniSection*      	LoadSection(const string& theSection);
		bool		        Save(ios_base::openmode openMode = ios::out|ios::trunc);
		bool		        UnLoad(){return Save();}
        bool                WasItFound(){return mWasFound;} //!Boolean indicating if the last key was found in the ini file
        bool                SectionExists(const string& section);

        //Reading
		// ReadValue: Default access method. Returns the raw string value
		string		        ReadValue(const string& Key, const string& Section = rr::gEmptyString);
 		string		        ReadString(const string& Key, const string& Section = rr::gEmptyString, const string& def_val=rr::gEmptyString);
		double		        ReadDouble(const string& Key, const string& Section = rr::gEmptyString, double def_value = 0);
		float		        ReadFloat(const string& Key, const string& Section = rr::gEmptyString, double def_value = 0){return (float) ReadDouble(Key, Section, def_value);}
		int			        ReadInteger(const string& Key, const string& Section = rr::gEmptyString, int def_value = 0);
		bool		        ReadBool(const string& Key, const string& Section = rr::gEmptyString, bool def_value = false);

       	//Writing
 		bool		        WriteValue(const string& Key, const string& Value, const string& Comment = rr::gEmptyString, const string& Section = rr::gEmptyString);
		bool		        WriteFloat(const string& Key, double value, const string& Comment = rr::gEmptyString, const string& Section = rr::gEmptyString);
		bool		        WriteInteger(const string& Key, int nValue, const string& Comment = rr::gEmptyString, const string& Section = rr::gEmptyString);
		bool		        WriteBool(const string& Key, bool bValue, const string& Comment = rr::gEmptyString, const string& Section = rr::gEmptyString);
		bool		        WriteString(const string& Key, const string& Value, const string& Comment = rr::gEmptyString, const string& Section = rr::gEmptyString){return WriteValue(Key, Value, Comment, Section);}
 		bool		        WriteNonKey(const string& nonKey, const string& Section = rr::gEmptyString);

    	//KeyHandling
		bool		        SetKeyComment(const string& Key, const string& Comment, const string& Section = rr::gEmptyString);
		bool		        SetSectionComment(const string& Section, const string& Comment);
		bool		        DeleteKey(const string& Key, const string& FromSection = rr::gEmptyString);
		bool		        DeleteSection(const string& Section);
		bool		        DeleteSectionsWithKeyValue(const string& key, const string& value);
		bool		        CreateKey(const string& Key, const string& Value, const string& Comment = rr::gEmptyString, const string& Section = rr::gEmptyString);

		//Section stuff
		bool		        CreateSection(const string& Section, const string& Comment = rr::gEmptyString);
		bool		        CreateSection(const string& Section, const string& Comment, KeyList Keys);

		// Utility Methods
		int				    SectionCount();
		int				    KeyCount();
        int 			    KeyCount(const string& section);
		void			    Clear();
		void			    SetFileName(const string& fName);
		string			    CommentStr(string& Comment);
		rrIniKey*		    GetKey(const string& Key, const string& Section);
		rrIniSection*	    GetSection(const string& Section, bool create = false);
		rrIniSection*	    GetSection(const unsigned int secNr);
        bool 			    ClearSection(const string& section);
        bool 			    IsModified(){return mIsDirty;}
};

}
#endif
