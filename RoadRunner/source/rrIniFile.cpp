#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <float.h>
#include <stdarg.h>
#include <string.h>
#include "rrUtils.h"
#include "rrStringUtils.h"
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrIniFile.h"

#undef CreateFile

using namespace rr;

namespace rr
{
//const bool IniFile::mAutoCreateSections = true;
//const bool IniFile::mAutoCreateKeys		= true;

IniFile::IniFile(const string& szFileName, bool autoLoad, bool autoSave)
:
mCommentIndicators(";#"),
mEqualIndicator("="),
mWhiteSpace(" \t\n\r"),
mIniFileName(szFileName),
mIsDirty(false),
mAutoSave(autoSave),
mAutoCreateKeys(1),
mAutoCreateSections(1)
{
	//mFlags = (mAutoCreateSections | mAutoCreateKeys);
	if(mIniFileName.size() > 2 && autoLoad == true)
	{
		Load();	//Load all sections and keys upon creation...
		Log(lDebug3)<<"Loaded file: " << mIniFileName.Get();
	}
}

IniFile::~IniFile()
{
	if (mIsDirty && mAutoSave)
    {
		Save();
    }

	Clear();
}

// Resets the member variables to their defaults
void IniFile::Clear()
{
	mIsDirty = true;			//A save after clear, will clear the phuysica; file
	for(unsigned int i = 0; i < mSections.size(); i++)
	{
		rrIniSection* section = mSections[i];
		delete section;
	}
	mSections.clear();
}

bool IniFile::SetFilePath(const string& path)
{
	return mIniFileName.SetPath(path);
}

bool IniFile::ClearSection(const string& aSection)
{
	rrIniSection* section = GetSection(aSection);
	if(section)
	{
		section->Clear();
		return true;
	}
	return false;
}

// Set's the mFileName member variable. For use when creating the IniFile
// object by hand versus loading it from a file
void IniFile::SetFileName(const string& FileName)
{
	if (mIniFileName.size() != 0 && CompareNoCase(FileName, mIniFileName) != 0)
	{
		mIsDirty = true;
	}

	mIniFileName = FileName;
}

// Attempts to load in the text file. If successful it will populate the
// Section list with the key/value pairs found in the file. Note that comments
// are saved so that they can be rewritten to the file later.
bool IniFile::Load(const string& newfName)
{
	if(newfName.size())
    {
    	mIniFileName = newfName;
    }

	if(mIniFileName.size() < 1)
	{
									Log(lError)<<"Ini file name is not set!";
		return false;
	}
	else
	{
									Log(lDebug3)<<"Loading IniFile: " <<mIniFileName.Get();
	}


    if(!FileExists(mIniFileName))
    {
									Log(lDebug2)<<"The ini file: \"" <<mIniFileName.Get()<<"\" does not exist. It will be created";
        if(!CreateFile(mIniFileName))
        {
    								Log(lError)<<"Failed to create ini file.. returning..";
            return false;                                    
        }
	}

	string fName(mIniFileName);
    fstream file;
	file.open(fName.c_str(), ios::in|ios::out);
	if(file.fail())
    {
    								Log(lError)<<"Opening file caused failbit to be set";
    }

	if (!file.is_open() )
	{
									Log(lError)<<"The ini file: '" <<mIniFileName<<"' could not be opened";
		return false;
	}
	else
	{
		bool bDone = false;
		bool bAutoKey = (mFlags & mAutoCreateKeys) 		== mAutoCreateKeys;
		bool bAutoSec = (mFlags & mAutoCreateSections) 	== mAutoCreateSections;


		rrIniSection* pSection = GetSection("");

		// These need to be set, we'll restore the original values later.
		mFlags |= mAutoCreateKeys;
		mFlags |= mAutoCreateSections;

		char* buffer = new char[MAX_LINE_BUFFER_SIZE];
        int lines = 0;
		while(!bDone)
		{
			string Comment;
			memset(buffer, 0, MAX_LINE_BUFFER_SIZE);
			file.getline(buffer, MAX_LINE_BUFFER_SIZE);
			lines++;

			string Line = buffer;
			Trim(Line);

			bDone = ( file.eof() || file.bad() || file.fail() );

			if ( Line.find_first_of(mCommentIndicators) == 0 )
			{
				Comment += "\n";
				Comment += Line;
			}
			else if ( Line.find_first_of('[') == 0 ) // Found a section
			{
				Line.erase( 0, 1 );
				Line.erase( Line.find_last_of(']'), 1 );
				pSection = GetSection(Line, true);
            	Log(lDebug3)<<"Located section: " + pSection->mName;
				Comment = string("");
			}
			else if ( Line.size() > 0 ) // we have a key, add this key/value pair
			{
                string Comment = "";
                string Key = "";
                string Value = "";

            	vector<string> record = SplitString(Line, mEqualIndicator + mCommentIndicators);
                if(record.size() > 1)
				{
	                string Comment = "";
                	string Key = Trim(record[0]);
                    string Value = Trim(record[1]);
                    if(record.size() > 2)
                    {
                    	Comment = record[2];
                    }

					if(pSection)
					{
						WriteValue(Key, Value, Comment, pSection->mName);
						Log(lDebug5)<<Key << " = "  <<Value;
					}
					else
					{
                		Log(lDebug5)<<Key << " = " << Value;
						Log(lWarning)<<"No section for key"<<Key<<". Key was ignored..";
					}
					Comment = string("");
				}
				else	//Not a valid Key. Store as a NonKey
				{
					if(pSection)
					{
						WriteNonKey(Line, pSection->mName);
					}
					Log(lDebug5)<<"Read NonKey on line: " << lines;
				}
			}
		}

        delete [] buffer;
		// Restore the original flag values.
		if(!bAutoKey)
        {
			mFlags &= ~mAutoCreateKeys;
        }

		if(!bAutoSec)
        {
			mFlags &= ~mAutoCreateSections;
        }
	}

	file.close();
	return true;
}

rrIniSection* IniFile::LoadSection(const string& theSection)
{
    rrIniSection* pSection = NULL;
	if(mIniFileName.size() < 1)
	{
									Log(lError)<<"Ini file name is not set!";
		return NULL;
	}
	else
	{
									Log(lDebug3)<<"Loading IniFile: " <<mIniFileName.Get();
	}

    fstream file;
    if(!FileExists(mIniFileName))
    {
									Log(lDebug2)<<"The ini file: \"" <<mIniFileName.Get()<<"\" does not exist. It will be created";
	}

	string fName(mIniFileName);
	file.open(fName.c_str(), ios::in|ios::out);
	if(file.fail())
    {
    								Log(lError)<<"Opening file caused failbit to be set";
    }

	if (!file.is_open() )
	{
									Log(lError)<<"The ini file: '" <<mIniFileName<<"' could not be opened";
		return NULL;
	}
	else
	{
		bool bDone = false;
		bool bAutoKey = (mFlags & mAutoCreateKeys) 		== mAutoCreateKeys;
		bool bAutoSec = (mFlags & mAutoCreateSections) 	== mAutoCreateSections;

		char* buffer = new char[MAX_LINE_BUFFER_SIZE];
		pSection = GetSection("");

		// These need to be set, we'll restore the original values later.
		mFlags |= mAutoCreateKeys;
		mFlags |= mAutoCreateSections;

		while(!bDone)
		{
			string Comment;
			memset(buffer, 0, MAX_LINE_BUFFER_SIZE);
			file.getline(buffer, MAX_LINE_BUFFER_SIZE);

			string Line = buffer;
			Trim(Line);

			bDone = ( file.eof() || file.bad() || file.fail() );

			if ( Line.find_first_of(mCommentIndicators) == 0 )
			{
				Comment += "\n";
				Comment += Line;
			}
			else
			if ( Line.find_first_of('[') == 0 ) // Found a section
			{
				Line.erase( 0, 1 );
				Line.erase( Line.find_last_of(']'), 1 );

                if(theSection == Line)
                {
    				CreateSection(Line, Comment);
	    			pSection = GetSection(Line);
										Log(lDebug3)<<"Located ini section: " + pSection->mName;
				    Comment = string("");
                    bDone = true;
                }
			}
			else if ( Line.size() > 0 ) // we have a key, add this key/value pair
			{
				string Key = GetNextWord(Line);
				string Value = Line;

				if ( Key.size() > 0 && Value.size() > 0 )
				{
					if(pSection)
					{
						WriteValue(Key, Value, Comment, pSection->mName);
										Log(lDebug5)<<"Read Key " + Key + " Value = " + Value;
					}
					else
					{
										Log(lDebug5)<<"Read Key " + Key + " Value = " + Value;
										Log(lWarning)<<"No section for key" + Key + ". Key was ignored..";
					}
					Comment = string("");
				}
				else	//Not a valid Key. Store as a NonKey
				{
					if(pSection)
					{
						WriteNonKey(Key, pSection->mName);
						Log(lDebug5)<<"Read a NonKey: " + Key;
					}
				}
			}
		}

		// Restore the original flag values.
		if (!bAutoKey)
		{
			mFlags &= ~mAutoCreateKeys;
		}

		if (!bAutoSec)
		{
			mFlags &= ~mAutoCreateSections;
		}
	}

	file.close();
	return pSection;
}

// Attempts to save the Section list and keys to the file. Note that if Load
// was never called (the IniFile object was created manually), then you
// must set the mFileName variable before calling save.
bool IniFile::Save(ios_base::openmode openMode)
{
	if ( mIniFileName.size() == 0 )
	{
		Log(lError)<<"No filename has been set. Can't save..";
		return false;
	}

    string aFName = mIniFileName.Get();
	fstream aFstream(aFName.c_str(), openMode);

	if(aFstream.is_open())
	{
		SectionItor s_pos;
		KeyItor k_pos;
		rrIniSection* Section;
		rrIniKey* Key;

		for (s_pos = mSections.begin(); s_pos != mSections.end(); s_pos++)
		{
			Section = (*s_pos);
			bool bWroteComment = false;

			if ( Section->mComment.size() > 0 )
			{
				bWroteComment = true;
				WriteLine(aFstream, "\n%s", CommentStr(Section->mComment).c_str());
			}

			if ( Section->mName.size() > 0 )
			{
				WriteLine(aFstream, "%s[%s]",
						bWroteComment ? "" : "\n",
						Section->mName.c_str());
			}

			for (k_pos = Section->mKeys.begin(); k_pos != Section->mKeys.end(); k_pos++)
			{
				Key = (*k_pos);

				if ( Key->mKey.size() > 0 && Key->mValue.size() > 0 )
				{
						WriteLine(aFstream, "%s%s%s%s%c%s",
						Key->mComment.size() > 0 ? "\n" : "",
						CommentStr(Key->mComment).c_str(),
						Key->mComment.size() > 0 ? "\n" : "",
						Key->mKey.c_str(),
						mEqualIndicator[0],
						Key->mValue.c_str());
				}
			}
		}
	}
	else
	{
		Log(lError)<<"[IniFile::Save] Unable to save file.";
		return false;
	}

	mIsDirty = false;
	aFstream.flush();
	aFstream.close();
								Log(lDebug3)<<"IniFile was saved";
	return true;
}

// Set the comment of a given key. Returns true if the key is not found.
bool IniFile::SetKeyComment(const string& mKey, const string& mComment, const string& szSection)
{
	KeyItor k_pos;
	rrIniSection* pSection;

	if ( (pSection = GetSection(szSection)) == NULL )
		return false;

	for (k_pos = pSection->mKeys.begin(); k_pos != pSection->mKeys.end(); k_pos++)
	{
		if ( CompareNoCase( (*k_pos)->mKey, mKey ) == 0 )
		{
			(*k_pos)->mComment = mComment;
			mIsDirty = true;
			return true;
		}
	}

	return false;
}

// Set the comment for a given section. Returns false if the section
// was not found.
bool IniFile::SetSectionComment(const string& Section, const string& Comment)
{
	SectionItor s_pos;

	for (s_pos = mSections.begin(); s_pos != mSections.end(); s_pos++)
	{
    	string name = (*s_pos)->mName;
		if ( CompareNoCase( name , Section ) == 0 )
		{
		    (*s_pos)->mComment = Comment;
			mIsDirty = true;
			return true;
		}
	}

	return false;
}

bool IniFile::SectionExists(const string& section)
{
	SectionItor s_pos;
	rrIniSection* Section;

    for (s_pos = mSections.begin(); s_pos != mSections.end(); s_pos++)
	{
			Section = (*s_pos);
            if(Section->mName == section)
            	return true;
    }
	return false;
}

// Given a key, a value and a section, this function will attempt to locate the
// Key within the given section, and if it finds it, change the keys value to
// the new value. If it does not locate the key, it will create a new key with
// the proper value and place it in the section requested.
bool IniFile::WriteValue(const string& mKey, const string& mValue, const string& mComment, const string& szSection)
{
	rrIniKey* 		pKey 		= GetKey(mKey, szSection);
	rrIniSection* 	pSection 	= GetSection(szSection);

	if (pSection == NULL)
	{
		if ( !(mFlags & mAutoCreateSections) || !CreateSection(szSection))
			return false;

		pSection = GetSection(szSection);
	}

	// Sanity check...
	if ( pSection == NULL )
		return false;

	// if the key does not exist in that section, and the value passed
	// is not string("") then add the new key.
	if ( pKey == NULL && mValue.size() > 0 && (mFlags & mAutoCreateKeys))
	{
		pKey = new rrIniKey;

		pKey->mKey = mKey;
		pKey->mValue = mValue;
		pKey->mComment = mComment;
		mIsDirty = true;
		pSection->mKeys.push_back(pKey);
		return true;
	}

	if ( pKey != NULL )
	{
		pKey->mValue = mValue;
		pKey->mComment = mComment;
		mIsDirty = true;
		return true;
	}

	return false;
}

bool IniFile::WriteNonKey(const string& nonKey, const string& section)
{
	rrIniSection* 	pSection 	= GetSection(section);
	if (pSection == NULL)
	{
		if ( !(mFlags & mAutoCreateSections) || !CreateSection(section))
			return false;

		pSection = GetSection(section);
	}

	// Sanity check...
	if (pSection == NULL )
		return false;

    pSection->mNonKeys.push_back(nonKey);
	return true;
}

// Passes the given float to WriteValue as a string
bool IniFile::WriteFloat(const string& mKey, double value, const string& mComment, const string& szSection)
{
	char szStr[64];
	snprintf(szStr, 64, "%g", value);
	return WriteValue(mKey, szStr, mComment, szSection);
}

// Passes the given int to WriteValue as a string
bool IniFile::WriteInteger(const string& mKey, int nValue, const string& mComment, const string& szSection)
{
	char szStr[64];
	snprintf(szStr, 64, "%d", nValue);
	return WriteValue(mKey, szStr, mComment, szSection);
}

// Passes the given bool to WriteValue as a string
bool IniFile::WriteBool(const string& mKey, bool bValue, const string& mComment, const string& szSection)
{
	string mValue = bValue ?  "true" : "false";

	return WriteValue(mKey, mValue, mComment, szSection);
}

// Returns the key value as a string object. A return value of
// string("") indicates that the key could not be found.
string IniFile::ReadValue(const string& mKey, const string& szSection)
{
	rrIniKey* pKey = GetKey(mKey, szSection);
  	mWasFound = pKey ? true : false;

    string value;
    if(pKey)
    {
        value = pKey->mValue;
    }
    else
    {
    	value = "";
    }

	return value;
}

// Returns the key value as a string object. A return value of
// string("") indicates that the key could not be found.
string IniFile::ReadString(const string& mKey, const string& szSection, const string& def_val)
{
	string 	str =  ReadValue(mKey, szSection);
	if(mWasFound)
	{
		return str;
	}
	return def_val;
}

// Returns the key value as a double type. Returns FLT_MIN if the key is
// not found.
double IniFile::ReadDouble(const string& mKey, const string& szSection, double def_value)
{
	string mValue = ReadValue(mKey, szSection);

	if(mWasFound)
	{
		return atof( mValue.c_str() );
	}

	return def_value;
}

// Returns the key value as an integer type. Returns INT_MIN if the key is
// not found.
int	IniFile::ReadInteger(const string& mKey, const string& szSection, int def_val)
{
	string mValue = ReadValue(mKey, szSection);

	if(mWasFound)
	{
		return	atoi(mValue.c_str());
	}
	return def_val;
}

// Returns the key value as a bool type. Returns false if the key is
// not found.
bool IniFile::ReadBool(const string& mKey, const string& szSection, bool def_value)
{
	bool bValue = def_value;
	string mValue = ReadValue(mKey, szSection);

	if(mWasFound)
	{
		if ( mValue.find("1") 						== 0 	
				|| CompareNoCase(mValue, "true") 	== 0 
				|| CompareNoCase(mValue, "yes") 	== 0 )
		{
			bValue = true;
		}
        else
        {
        	bValue = false;
        }
	}	

	return bValue;
}

// Delete a specific section. Returns false if the section cannot be
// found or true when sucessfully deleted.
bool IniFile::DeleteSection(const string& Section)
{
	SectionItor s_pos;

	for (s_pos = mSections.begin(); s_pos != mSections.end(); s_pos++)
	{
		if ( CompareNoCase( (*s_pos)->mName, Section ) == 0 )
		{
        	//Found a section..
            rrIniSection *aSection =  (*s_pos);
			mSections.erase(s_pos);
            delete (aSection);
			return true;
		}
	}

	return false;
}

bool IniFile::DeleteSectionsWithKeyValue(const string& keyName, const string& value)
{
	//First find sections containing the key
	SectionItor s_pos = mSections.begin();
    s_pos = mSections.begin();
	while (s_pos != mSections.end())
	{
		rrIniKey* key = GetKey(keyName, (*s_pos)->mName);
        {
        	if(key && key->mValue == value)
            {
           		//remove section
              	DeleteSection((*s_pos)->mName);
			    s_pos = mSections.begin();
            }
            else
            {
            	++s_pos;
            }
        }
    }
	return true;
}

// Delete a specific key in a specific section. Returns false if the key
// cannot be found or true when sucessfully deleted.
bool IniFile::DeleteKey(const string& Key, const string& FromSection)
{
	KeyItor k_pos;
	rrIniSection* pSection;

	if ( (pSection = GetSection(FromSection)) == NULL )
		return false;

	for (k_pos = pSection->mKeys.begin(); k_pos != pSection->mKeys.end(); k_pos++)
	{
		if ( CompareNoCase( (*k_pos)->mKey, Key ) == 0 )
		{
        	rrIniKey* aKey = (*k_pos);
			pSection->mKeys.erase(k_pos);
            delete aKey;
			return true;
		}
	}

	return false;
}

// Given a key, a value and a section, this function will attempt to locate the
// Key within the given section, and if it finds it, change the keys value to
// the new value. If it does not locate the key, it will create a new key with
// the proper value and place it in the section requested.
bool IniFile::CreateKey(const string& mKey, const string& mValue, const string& mComment, const string& szSection)
{
	bool bAutoKey = (mFlags & mAutoCreateKeys) == mAutoCreateKeys;
	bool bReturn  = false;

	mFlags |= mAutoCreateKeys;

	bReturn = WriteValue(mKey, mValue, mComment, szSection);

	if (!bAutoKey)
	{
		mFlags &= ~mAutoCreateKeys;
	}

	return bReturn;
}

// Given a section name, this function first checks to see if the given section
// allready exists in the list or not, if not, it creates the new section and
// assigns it the comment given in mComment.  The function returns true if
// sucessfully created, or false otherwise.
bool IniFile::CreateSection(const string& Section, const string& mComment)
{
	rrIniSection* pSection = GetSection(Section);

	if ( pSection )
	{
		Log(lDebug5)<<"[IniFile::CreateSection] Section "<<Section.c_str()<<" already exists. Aborting.";
		return false;
	}

	pSection = new rrIniSection;
	pSection->mName = Section;
	pSection->mComment = mComment;
	mSections.push_back(pSection);
	mIsDirty = true;
	return true;
}

// Given a section name, this function first checks to see if the given section
// allready exists in the list or not, if not, it creates the new section and
// assigns it the comment given in mComment.  The function returns true if
// sucessfully created, or false otherwise. This version accpets a KeyList
// and Writes up the newly created Section with the keys in the list.
bool IniFile::CreateSection(const string& Section, const string& Comment, KeyList Keys)
{
	if ( !CreateSection(Section, Comment) )
		return false;

	rrIniSection* pSection = GetSection(Section);

	if ( !pSection )
		return false;

	KeyItor k_pos;

	pSection->mName = Section;
	for (k_pos = Keys.begin(); k_pos != Keys.end(); k_pos++)
	{
		rrIniKey* pKey = new rrIniKey;
		pKey->mComment = (*k_pos)->mComment;
		pKey->mKey = (*k_pos)->mKey;
		pKey->mValue = (*k_pos)->mValue;
		pSection->mKeys.push_back(pKey);
	}

	mSections.push_back(pSection);
	mIsDirty = true;

	return true;
}

// Simply returns the number of sections in the list.
int IniFile::SectionCount()
{
	return mSections.size();
}

// Returns the total number of keys contained within all the sections.
int IniFile::KeyCount()
{
	int nCounter = 0;
	SectionItor s_pos;

	for (s_pos = mSections.begin(); s_pos != mSections.end(); s_pos++)
		nCounter += (*s_pos)->mKeys.size();

	return nCounter;
}

int IniFile::KeyCount(const string& section)
{
	//Get the section
    rrIniSection* iniSection = GetSection(section);
    return iniSection->KeyCount();
}

// Given a key and section name, looks up the key and if found, returns a
// pointer to that key, otherwise returns NULL.
rrIniKey*	IniFile::GetKey(const string& Key, const string& Section)
{
	rrIniSection* pSection;
	KeyItor k_pos;

	// Since our default section has a name value of string("") this should
	// always return a valid section, wether or not it has any keys in it is
	// another matter.
	if ( (pSection = GetSection(Section)) == NULL )
		return NULL;

	for (k_pos = pSection->mKeys.begin(); k_pos != pSection->mKeys.end(); k_pos++)
	{
		if ( CompareNoCase( (*k_pos)->mKey, Key ) == 0 )
			return (*k_pos);
	}

	return NULL;
}

// Given a section name, locates that section in the list and returns a pointer
// to it. If the section was not found, returns NULL
rrIniSection* IniFile::GetSection(const string& Section, bool create)
{
	SectionItor s_pos;
	for (s_pos = mSections.begin(); s_pos != mSections.end(); s_pos++)
	{
    	if(s_pos == mSections.end())
        {
        	return NULL;
        }
        string secName = (*s_pos)->mName;
		if ( CompareNoCase( secName, Section ) == 0 )
        {
        	rrIniSection* sec = (*s_pos);
			return sec;
        }
	}

    if(create)
    {
        CreateSection(Section);
        return GetSection(Section, false);
    }

	return NULL;
}

rrIniSection* IniFile::GetSection(const unsigned int sectionNr)
{
	if(sectionNr < mSections.size() && mSections.size() > 0)
    	return mSections[sectionNr];

	return NULL;
}

string IniFile::CommentStr(string& mComment)
{
	string szNewStr = string("");

	Trim(mComment);

		if ( mComment.size() == 0 )
		  return mComment;


	if ( mComment.find_first_of(mCommentIndicators) != 0 )
	{
		szNewStr = mCommentIndicators[0];
		szNewStr += " ";
	}

	szNewStr += mComment;

	return szNewStr;
}

// Given a key+delimiter+ value string, pulls the key name from the string,
// deletes the delimiter and alters the original string to contain the
// remainder.  Returns the key
string IniFile::GetNextWord(string& CommandLine)
{
	int nPos = CommandLine.find_first_of(mEqualIndicator);
	string sWord = string("");

	if ( nPos > -1 )
	{
		sWord = CommandLine.substr(0, nPos);
		CommandLine.erase(0, nPos+1);
	}
	else
	{
		sWord = CommandLine;
		CommandLine = string("");
	}

	Trim(sWord);
	return sWord;
}

string IniFile::Trim(string& str)
{
	string szTrimChars 	= mWhiteSpace;
	szTrimChars 		+= mEqualIndicator;

	// Trim Both leading and trailing spaces
	size_t startpos = str.find_first_not_of(szTrimChars); 	// Find the first character position after excluding leading blank spaces
	size_t endpos 	= str.find_last_not_of(szTrimChars); 	// Find the first character position from reverse af

	// if all spaces or empty return an empty string
	if(( string::npos == startpos ) || ( string::npos == endpos))
	{
		str = "";
	}
	else
    {
		str = str.substr(startpos, endpos-startpos + 1);
    }

	return str;
}

// Writes the formatted output to the file stream, returning the number of
// bytes written.
int IniFile::WriteLine(fstream& stream, const char* fmt, ...)
{
	char *buf = new char[MAX_LINE_BUFFER_SIZE];
	int nLength;
	string szMsg;

	memset(buf, 0, MAX_LINE_BUFFER_SIZE);
	va_list args;

	va_start (args, fmt);
	  nLength = vsnprintf(buf, MAX_LINE_BUFFER_SIZE, fmt, args);
	va_end (args);


	if ( buf[nLength] != '\n' && buf[nLength] != '\r' )
		buf[nLength++] = '\n';

	stream.write(buf, nLength);

	delete [] buf;
	return nLength;
}


}
