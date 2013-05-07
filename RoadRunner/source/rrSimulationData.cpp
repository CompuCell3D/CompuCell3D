#ifdef USE_PCH
#include "rr_pch.h"
#endif
#pragma hdrstop
#include <iomanip>
#include "rrLogger.h"
#include "rrUtils.h"
#include "rrStringUtils.h"
#include "rrSimulationData.h"
//---------------------------------------------------------------------------
using namespace std;

namespace rr
{

SimulationData::SimulationData()
:
mTimePrecision(6),
mDataPrecision(16)
{}

SimulationData::SimulationData(const StringList& colNames, const DoubleMatrix& theData)
:
mColumnNames(colNames),
mTheData(theData)
{}

int SimulationData::cSize() const
{
    return mTheData.CSize();
}

int SimulationData::rSize() const
{
    return mTheData.RSize();
}

void SimulationData::setName(const string& name)
{
    mName = name;
//    mTheData.SetNamePointer(&mName);
}

SimulationData& SimulationData::operator= (const SimulationData& rhs)
{
	if(this == &rhs)
    {
    	return *this;
    }

    mTheData = rhs.mTheData;
    mColumnNames = rhs.mColumnNames;

    return *this;

}

bool SimulationData::append(const SimulationData& data)
{
	//When appending data, the number of rows have to match with current data
    if(mTheData.RSize() > 0)
    {
		if(data.rSize() != rSize())
        {
        	return false;
        }
	}
    else
    {
    	(*this) = data;
        return true;
    }

    int currColSize = cSize();

    SimulationData temp(mColumnNames, mTheData);

    int newCSize = cSize() + data.cSize();
    mTheData.resize(data.rSize(), newCSize );

	for(int row = 0; row < temp.rSize(); row++)
    {
    	for( int col = 0; col < temp.cSize(); col++)
        {
        	mTheData(row, col) = temp(row, col);
        }
    }


    for(int row = 0; row < mTheData.RSize(); row++)
    {
        for(int col = 0; col < data.cSize(); col++)
        {
            mTheData(row, col + currColSize) = data(row, col);
        }
    }


    for(int col = 0; col < data.cSize(); col++)
    {
    	mColumnNames.Append(data.getColumnName(col));
    }
	return true;
}

StringList SimulationData::getColumnNames() const
{
    return mColumnNames;
}

string SimulationData::getColumnName(const int& col) const
{
	if(col < mColumnNames.Count())
    {
		return mColumnNames[col];
    }

    return "Bad Column..";
}

pair<int,int> SimulationData::dimension() const
{
    return pair<int,int>(mTheData.RSize(), mTheData.CSize());
}

string SimulationData::getName() const
{
    return mName;
}

void SimulationData::setTimeDataPrecision(const int& prec)
{
    mTimePrecision = prec;
}

void SimulationData::setDataPrecision(const int& prec)
{
    mDataPrecision = prec;
}

string SimulationData::getColumnNamesAsString() const
{
    return mColumnNames.AsString();
}

void SimulationData::allocate(const int& cSize, const int& rSize)
{
    mTheData.Allocate(cSize, rSize);
}

//=========== OPERATORS
double& SimulationData::operator() (const unsigned& row, const unsigned& col)
{
    return mTheData(row,col);
}

double SimulationData::operator() (const unsigned& row, const unsigned& col) const
{
    return mTheData(row,col);
}

void SimulationData::setColumnNames(const StringList& colNames)
{
    mColumnNames = colNames;
    Log(lDebug3)<<"Simulation Data Columns: "<<mColumnNames;
}

void SimulationData::setNrOfCols(const int& cols)
{
    mTheData.Allocate(1, cols);
}

void SimulationData::setData(const DoubleMatrix& theData)
{
    mTheData = theData;
    Log(lDebug5)<<"Simulation Data =========== \n"<<mTheData;
    check();
}

bool SimulationData::check() const
{
    if(mTheData.CSize() != mColumnNames.Count())
    {
        Log(lError)<<"Number of columns ("<<mTheData.CSize()<<") in simulation data is not equal to number of columns in column header ("<<mColumnNames.Count()<<")";
        return false;
    }
    return true;
}

bool SimulationData::load(const string& fName)
{
    if(!FileExists(fName))
    {
        return false;
    }

    vector<string> lines = GetLinesInFile(fName.c_str());
    if(!lines.size())
    {
        Log(lError)<<"Failed reading/opening file "<<fName;
        return false;
    }

    mColumnNames = StringList(lines[0]);
    Log(lInfo)<<mColumnNames;

    mTheData.resize(lines.size() -1, mColumnNames.Count());

    for(int i = 0; i < mTheData.RSize(); i++)
    {
        StringList aLine(lines[i+1]);
        for(int j = 0; j < aLine.Count(); j++)
        {
            mTheData(i,j) = ToDouble(aLine[j]);
        }
    }

    return true;
}

bool SimulationData::writeTo(const string& fileName)
{
	ofstream aFile(fileName.c_str());
    if(!aFile)
    {
    	Log(lError)<<"Failed opening file: "<<fileName;
        return false;
    }
    aFile<<(*this);
    aFile.close();

    return true;
}

ostream& operator << (ostream& ss, const SimulationData& data)
{
    //Check that the dimensions of col header and data is ok
    if(!data.check())
    {
        Log(lError)<<"Can't write data..";
        return ss;
    }

    //First create the header
    for(u_int i = 0; i < data.mColumnNames.Count(); i++)
    {
        ss<<data.mColumnNames[i];
        if(i < data.mColumnNames.Count() - 1)
        {
            ss << ",";
        }
        else
        {
            ss << endl;
        }
    }
    //Then the data
    for(u_int row = 0; row < data.mTheData.RSize(); row++)
    {
        for(u_int col = 0; col < data.mTheData.CSize(); col++)
        {
            if(col == 0)
            {
                ss<<setprecision(data.mTimePrecision)<<data.mTheData(row, col);
            }
            else
            {
                ss<<setprecision(data.mDataPrecision)<<data.mTheData(row, col);
            }

            if(col <data.mTheData.CSize() -1)
            {
                ss << ",";
            }
            else
            {
                ss << endl;
            }
        }
    }

    return ss;
}

}//namespace
