#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include "rrStringUtils.h"
#include "rrFileName.h"

using std::ostream;

namespace rr
{
//---------------------------------------------------------------------------
FileName::FileName(const string& name, const string& path)
{
    SetNameAndPath(path, name);
}

FileName::~FileName()
{

}

FileName::FileName(const FileName& fN)
{
	mPath = fN.GetPath();
	mName = fN.GetFileName();
	MakeFileString();
}

FileName& FileName::operator=(const FileName& fN)
{
	mPath = fN.GetPath();
	mName = fN.GetFileName();
	MakeFileString();
	return *this;
}

FileName& FileName::operator=(const string& fN)
{
	mPathAndName = fN;
	mPath = ExtractFilePath(fN);
	mName = rr::ExtractFileName(fN);
	return *this;
}

bool FileName::operator!=(const char* fN)
{
	return (mName != fN) ? false : true;
}

string FileName::GetFileName() const {return mName;}
string FileName::GetPath() const {return mPath;}
string FileName::GetPathAndFileName() const {return mPathAndName;}
unsigned int FileName::size(){return mName.size();}

FileName::operator string() {return mPathAndName;}
string FileName::Get()
{
	MakeFileString();
	return mPathAndName;
}

string FileName::GetFileNameNoExtension()
{
    //remove extension
	string fname = ExtractFileNameNoExtension(GetFileName());
    return fname;
}

void FileName::SetFileName(const string& name)
{
	mName = name;
	MakeFileString();
}

bool FileName::SetPath(const string& path)
{
	mPath = path;
	MakeFileString();
    return true;
}

void FileName::SetNameAndPath(const string& path, const string& name)
{
    mPath = path;
    mName = name;
	MakeFileString();
}

void FileName::SetFileNameAndPath(const string& file)
{
	if(!file.size())
	{
		mPathAndName = "";
		mName = "";
		mPath = "";
	}
	else
	{
		mName = ExtractFileName(file);
		mPath = ExtractFilePath(file);
		MakeFileString();
	}
}

void FileName::MakeFileString()
{
	mPathAndName = "";
	if(mPath.size())
	{
    	if(mPath[mPath.size()-1] == '\\')
		{
			mPathAndName = mPath + mName;
        }
		else //Add slashes to path
		{
			//exit(-1);
			mPath = mPath + "\\";
			mPathAndName = mPath + mName;
    	}
	}
	else // No path
	{
    	if(mName.size())
		{
        	mPathAndName += mName;
		}
	}
}

ostream& operator <<(ostream& os, FileName& obj)
{
    os<<obj.Get();
    return os;
}

}
