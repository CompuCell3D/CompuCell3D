/*
#############################################################################
#                                                                           #
# BSD 3-Clause License (see https://opensource.org/licenses/BSD-3-Clause)   #
#                                                                           #
# Copyright (c) 2011-2020 Institut Curie, 26 rue d'Ulm, Paris, France       #
# All rights reserved.                                                      #
#                                                                           #
# Redistribution and use in source and binary forms, with or without        #
# modification, are permitted provided that the following conditions are    #
# met:                                                                      #
#                                                                           #
# 1. Redistributions of source code must retain the above copyright notice, #
# this list of conditions and the following disclaimer.                     #
#                                                                           #
# 2. Redistributions in binary form must reproduce the above copyright      #
# notice, this list of conditions and the following disclaimer in the       #
# documentation and/or other materials provided with the distribution.      #
#                                                                           #
# 3. Neither the name of the copyright holder nor the names of its          #
# contributors may be used to endorse or promote products derived from this #
# software without specific prior written permission.                       #
#                                                                           #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED #
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A           #
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER #
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  #
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,       #
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR        #
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    #
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      #
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        #
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              #
#                                                                           #
#############################################################################

   Module:
     FixedPointDisplayer.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     Decembre 2020
*/

#include "FixedPointDisplayer.h"
#include "BooleanNetwork.h"
#include "Utils.h"

void CSVFixedPointDisplayer::begin(size_t size) {
  os << "Fixed Points (" << size << ")\n";
  if (size > 0) {
    os << "FP\tProba\tState\t";
    network->displayHeader(os);
  }
}

void CSVFixedPointDisplayer::displayFixedPoint(size_t num, const NetworkState& state, unsigned int val, unsigned int sample_count) {
  os << "#" << num << "\t";
  if (hexfloat) {
    os << fmthexdouble((double)val / sample_count) <<  "\t";
  } else {
    os << ((double)val / sample_count) <<  "\t";
  }
  state.displayOneLine(os, network);
  os << '\t';
  state.display(os, network);
}

void CSVFixedPointDisplayer::end() {
}

void JsonFixedPointDisplayer::begin(size_t size) {
  os << "{\"count\":" << size << "\"points\":[";
}

void JsonFixedPointDisplayer::displayFixedPoint(size_t num, const NetworkState& state, unsigned int val, unsigned int sample_count) {
  os << "{\"num\":" << num << ",";
  os << "\"value\":";
  if (hexfloat) {
    os << fmthexdouble((double)val / sample_count, true) <<  "\t";
  } else {
    os << ((double)val / sample_count) <<  "\t";
  }
  os << ",\"state\":\"";
  state.displayOneLine(os, network);
  // TBD: missing
  //state.display(os, network);
  os << "\"}";
}

void JsonFixedPointDisplayer::end() {
  os << "]}";
}


