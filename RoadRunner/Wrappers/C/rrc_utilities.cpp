/**
 * @file utilities.cpp
 * @brief roadRunner C API 2012
 * @author Totte Karlsson & Herbert M Sauro
 *
 * <--------------------------------------------------------------
 * This file is part of cRoadRunner.
 * See http://code.google.com/p/roadrunnerlib for more details.
 *
 * Copyright (C) 2012-2013
 *   University of Washington, Seattle, WA, USA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * In plain english this means:
 *
 * You CAN freely download and use this software, in whole or in part, for personal,
 * company internal, or commercial purposes;
 *
 * You CAN use the software in packages or distributions that you create.
 *
 * You SHOULD include a copy of the license in any redistribution you may make;
 *
 * You are NOT required include the source of software, or of any modifications you may
 * have made to it, in any redistribution you may assemble that includes it.
 *
 * YOU CANNOT:
 *
 * redistribute any piece of this software without proper attribution;
*/

#pragma hdrstop
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#include "rrRoadRunner.h"
#include "rrException.h"
#include "rrc_utilities.h" 			// Need to include this before the support header..
#include "rrc_support.h"   //Support functions, not exposed as api functions and or data

//---------------------------------------------------------------------------

namespace rrc
{
using namespace std;
using namespace rr;
using namespace rrc;

char* rrCallConv getFileContent(const char* fName)
{
	try
    {
    	string fContent = GetFileContent(fName);
        return createText(fContent);
    }
    CATCH_PTR_MACRO

}


bool rrCallConv compileSource(RRHandle handle, const char* sourceFileName)
{
	try
    {
        RoadRunner* rri = castFrom(handle);
        return rri->compileSource(sourceFileName);
    }
    CATCH_BOOL_MACRO
}
#if defined(_WIN32)
int WINAPI DllEntryPoint(HINSTANCE hinst, unsigned long reason, void* lpReserved)
{
    return 1;
}
#endif


}

