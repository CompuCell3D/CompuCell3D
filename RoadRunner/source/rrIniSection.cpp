#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrUtils.h"
#include "rrStringUtils.h"
#include "rrLogger.h"
#include "rrIniSection.h"

using namespace rr;

namespace rr
{
rrIniSection::rrIniSection()
:
mIsDirty(true),
mName(gEmptyString),
mComment(gEmptyString)
{

}

rrIniSection::rrIniSection(const string& nameValueString, const char& sep)
:
mIsDirty(true),
mName(gEmptyString),
mComment(gEmptyString)
{
	vector<string> keys = SplitString(nameValueString, sep);

    //Insert each key in the section
    for(unsigned int i = 0; i < keys.size(); i++)
    {
        rrIniKey *aKey = new rrIniKey(keys[i]);
    	mKeys.push_back(aKey);
    }
}

rrIniSection::~rrIniSection()
{
	//detete all keys
    for(unsigned int i = 0; i < mKeys.size(); i++)
    {
    	rrIniKey *key = mKeys[i];
   		delete key;
    }
	mKeys.clear();
}

//IniKey function
rrIniKey*	rrIniSection::GetKey(const string& keyName, bool create)
{
	//Go trough the key list and return key with key name
   	KeyItor k_pos;
	for (k_pos = mKeys.begin(); k_pos != mKeys.end(); k_pos++)
	{
		if ( CompareNoCase( (*k_pos)->mKey, keyName ) == 0 )
			return *k_pos;
	}

    if(create)
    {
        CreateKey(keyName);
        return GetKey(keyName, false);
    }

	return NULL;
}

//IniKey function
rrIniKey*	rrIniSection::GetKey(const int& keyNr)
{
	//Go trough the key list and return key with key name

    if(keyNr < mKeys.size())
    {
    	return mKeys[keyNr];
    }

	return NULL;
}

string rrIniSection::GetNonKeysAsString()
{
	string tmp = "";
   	NonKeyItor listPos;

    if(!mNonKeys.size())
    {
        return tmp;
    }

    for(listPos = mNonKeys.begin(); listPos != mNonKeys.end(); listPos++)
	{
		if ((*listPos).size())
        {
			tmp += (*listPos);
            tmp += "\n";
        }
	}

	return tmp;
}

string rrIniSection::AsString()
{
	string tmp = "";
   	KeyItor listPos;
    for(listPos = mKeys.begin(); listPos != mKeys.end(); listPos++)
	{
		if ((*listPos)->AsString().size())
        {
			tmp += (*listPos)->AsString();
            tmp += "\n";
        }
	}

    return tmp;
}

rrIniKey* rrIniSection::CreateKey(const string& mKey, const string& mValue, const string& mComment)
{
    rrIniKey* 		pKey = GetKey(mKey);

    //Check if the key exists
    if(pKey)
    {
        pKey->mValue = mValue;
        pKey->mComment = mComment;
        return pKey;
    }

    //Create the key
    pKey = new rrIniKey;

    pKey->mKey = mKey;
    pKey->mValue = mValue;
    pKey->mComment = mComment;
    mIsDirty = true;
    mKeys.push_back(pKey);
	return pKey;
}

}//namespace
