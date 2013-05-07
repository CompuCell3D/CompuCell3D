#pragma hdrstop
#include <sstream>
#include <iomanip>
#include "rrUtils.h"
#include "rrPlugin.h"
#include "rrParameter.h"

//---------------------------------------------------------------------------
using namespace std;
namespace rr
{

Plugin::Plugin(const std::string& name, const std::string& cat, RoadRunner* aRR)
:
mName(name),
mAuthor("Totte Karlsson"),
mCategory(cat),
mVersion("0.1"),
mCopyright("Totte Karlsson, Herbert Sauro, Systems Biology, UW 2012"),
mRR(aRR)//,
//mCapability("PluginCapabilities", "", "")
{
}

Plugin::~Plugin()
{}

vector<string>& Plugin::getLog()
{
	return mLog;
}

bool Plugin::setParameter(const string& nameOf, void* value, Capability& capability)
{

	//Go trough the parameter container and look for parameter
    for(int i = 0; i < capability.nrOfParameters(); i++)
    {
        BaseParameter* aParameter = const_cast<BaseParameter*>( &(capability[i]) );

        if(dynamic_cast< Parameter<int>* >(aParameter))
        {
	        Parameter<int> *aIntPar = dynamic_cast< Parameter<int>* >(aParameter);
            int *aVal = reinterpret_cast<int*>(value);
        	aIntPar->setValue( *aVal);
            return true;
        }
    }
	return false;
}

bool Plugin::setParameter(const string& nameOf, const char* value, Capability& capability)
{
	//Go trough the parameter container and look for parameter
    for(int i = 0; i < capability.nrOfParameters(); i++)
    {
        BaseParameter* aParameter = const_cast<BaseParameter*>( &(capability[i]) );

        if(dynamic_cast< Parameter<int>* >(aParameter))
        {
	        Parameter<int> *aIntPar = dynamic_cast< Parameter<int>* >(aParameter);
            int aVal = rr::ToInt(value);
        	aIntPar->setValue( aVal);
            return true;
        }
    }
	return false;
}

bool Plugin::setParameter(const string& nameOf, const char* value)
{
	if(!mCapabilities.size())
    {
    	return false;
    }
    Capability& capability = mCapabilities[0];

	//Go trough the parameter container and look for parameter
    for(int i = 0; i < capability.nrOfParameters(); i++)
    {
        BaseParameter* aParameter = const_cast<BaseParameter*>( &(capability[i]) );

        if(dynamic_cast< Parameter<int>* >(aParameter))
        {
	        Parameter<int> *aIntPar = dynamic_cast< Parameter<int>* >(aParameter);
            int aVal = rr::ToInt(value);
        	aIntPar->setValue( aVal);
            return true;
        }
    }
	return false;
}

string Plugin::getName()
{
	return mName;
}

string Plugin::getAuthor()
{
	return mAuthor;
}

string Plugin::getCategory()
{
	return mCategory;
}

string Plugin::getVersion()
{
	return mVersion;
}

string Plugin::getCopyright()
{
	return mCopyright;
}

string Plugin::getInfo() //Obs. subclasses may over ride this function and add more info
{
    stringstream msg;

    msg<<setfill('.');
    msg<<setw(30)<<left<<"Name"<<mName<<"\n";
    msg<<setw(30)<<left<<"Author"<<mAuthor<<"\n";
    msg<<setw(30)<<left<<"Category"<<mCategory<<"\n";
    msg<<setw(30)<<left<<"Version"<<mVersion<<"\n";
    msg<<setw(30)<<left<<"Copyright"<<mCopyright<<"\n";

	msg<<"=== Capabilities ====\n";
    for(int i = 0; i < mCapabilities.size(); i++)
    {
    	msg<<mCapabilities[i];
    }
    return msg.str();
}

vector<Capability>*	 Plugin::getCapabilities()
{
	return &mCapabilities;
}

Parameters* Plugin::getParameters(const string& capName)
{
	//Return parameters for capability with name
    for(int i = 0; i < mCapabilities.size(); i++)
    {
        if(mCapabilities[i].getName() == capName)
        {
            return mCapabilities[i].getParameters();
        }
    }

	return NULL;
}

Parameters* Plugin::getParameters(Capability& capability)
{
	return capability.getParameters();
}

BaseParameter* Plugin::getParameter(const string& para)
{
	if(mCapabilities.size())
    {
		return mCapabilities[0].getParameter(para);
    }
    return NULL;
}

BaseParameter* Plugin::getParameter(const string& para, Capability& capability)
{
	return capability.getParameter(para);
}

Capability* Plugin::getCapability(const string& name)
{
	for(int i = 0; i < mCapabilities.size(); i++)
    {
		if(mCapabilities[i].getName() == name)
        {
        	return &(mCapabilities[i]);
        }
    }
    return NULL;
}

PluginLogger::PluginLogger(vector<string>* container)
:
mLogs(container)
{}

PluginLogger::~PluginLogger()
{
	vector<string> lines = rr::SplitString(mStream.str(),"\n");

    for(int i = 0; i < lines.size(); i++)
    {
	   mLogs->push_back(lines[i].c_str());

    }
}

std::ostringstream& PluginLogger::Get()
{
	return mStream;
}


}

