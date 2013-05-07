#include "XmlTestReporter.h"
#include "Config.h"

#include <iostream>
#include <sstream>
#include <string>
#include <time.h>

using std::string;
using std::ostringstream;
using std::ostream;

string ExtractFileName(const string& fName);
const string currentDateTime();
namespace {

void ReplaceChar(string& str, char c, string const& replacement)
{
    for (size_t pos = str.find(c); pos != string::npos; pos = str.find(c, pos + 1))
        str.replace(pos, 1, replacement);
}

string XmlEscape(string const& value)
{
    string escaped = value;

    ReplaceChar(escaped, '&', "&amp;");
    ReplaceChar(escaped, '<', "&lt;");
    ReplaceChar(escaped, '>', "&gt;");
    ReplaceChar(escaped, '\'', "&apos;");
    ReplaceChar(escaped, '\"', "&quot;");
 
    return escaped;
}

string BuildFailureMessage(string const& file, int line, string const& message)
{
    ostringstream failureMessage;
    failureMessage << file << "(" << line << ") : " << message;
    return failureMessage.str();
}

}

namespace UnitTest {

XmlTestReporter::XmlTestReporter(ostream& ostream)
    : m_ostream(ostream)
{
}

void XmlTestReporter::ReportSummary(int totalTestCount, int failedTestCount,
                                    int failureCount, float secondsElapsed)
{
    AddXmlElement(m_ostream, NULL);

    BeginResults(m_ostream, totalTestCount, failedTestCount, failureCount, secondsElapsed);

    DeferredTestResultList const& results = GetResults();
    for (DeferredTestResultList::const_iterator i = results.begin(); i != results.end(); ++i)
    {
        BeginTest(m_ostream, *i);

        if (i->failed)
            AddFailure(m_ostream, *i);

        EndTest(m_ostream, *i);
    }

    EndResults(m_ostream);
}

void XmlTestReporter::AddXmlElement(ostream& os, char const* encoding)
{
    os << "<?xml version=\"1.0\"";

    if (encoding != NULL)
        os << " encoding=\"" << encoding << "\"";
	else
    	os << " encoding=\"UTF-8\"";

    os << "?>\n";
    os<<"<?xml-stylesheet href=\"api_tests.xsl\" type=\"text/xsl\" ?>\n";
}

void XmlTestReporter::BeginResults(std::ostream& os, int totalTestCount, int failedTestCount,
                                   int failureCount, float secondsElapsed)
{
   os << "<unittest-results"
   	   << " date_time = \""<< currentDateTime() <<  "\""
       << " tests=\"" << totalTestCount << "\""
       << " failedtests=\"" << failedTestCount << "\""
       << " failures=\"" << failureCount << "\""
       << " time=\"" << secondsElapsed << "\""
       << ">\n";
}

void XmlTestReporter::EndResults(std::ostream& os)
{
    os << "</unittest-results>\n";
}

void XmlTestReporter::BeginTest(std::ostream& os, DeferredTestResult const& result)
{
    os << "<test"
        << " suite=\"" << result.suiteName << "\""
        << " name=\"" << result.testName << "\""
        << " time=\"" << result.timeElapsed << "\"";
}

void XmlTestReporter::EndTest(std::ostream& os, DeferredTestResult const& result)
{
    if (result.failed)
        os << "</test>";
    else
        os << "/>";

	os<<"\n";
}

void XmlTestReporter::AddFailure(std::ostream& os, DeferredTestResult const& result)
{
    os << ">"; // close <test> element

    for (DeferredTestResult::FailureVec::const_iterator it = result.failures.begin(); 
         it != result.failures.end(); 
         ++it)
    {
        string const escapedMessage = XmlEscape(it->second);
        string fileNoPath = ExtractFileName(result.failureFile);
        string const message = BuildFailureMessage(fileNoPath, it->first, escapedMessage);

        os << "<failure" << " message=\"" << message << "\"" << "/>";
    }
}
}//End of namespace

string ExtractFileName(const string& fileN)
{
    string fName;
    if(fileN.find_last_of( '\\' ) != std::string::npos)
    {
        fName = fileN.substr(fileN.find_last_of( '\\' )+ 1, fileN.size());
        return fName;
    }
    else if(fileN.find_last_of( '/' ) != std::string::npos)
    {
        fName = fileN.substr(fileN.find_last_of( '/' ) + 1, fileN.size());
        return fName;
    }

    return fileN; //There was no path in present..
}

const string currentDateTime()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://www.cplusplus.com/reference/clibrary/ctime/strftime/
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

    return string(buf);
}
