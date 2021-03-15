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
     ProjTrajDisplayer.cc

   Authors:
     Eric Viara <viara@sysra.com>
     Gautier Stoll <gautier.stoll@curie.fr>
     Vincent Noël <vincent.noel@curie.fr>
 
   Date:
     December 2020
*/

#include "BooleanNetwork.h"
#include "ProbTrajDisplayer.h"
#include "Utils.h"
#include <iomanip>

void JSONProbTrajDisplayer::beginDisplay() {
  os_probtraj << '[';
}

void JSONProbTrajDisplayer::beginTimeTickDisplay() {
  if (current_line > 0) {
    os_probtraj << ',';
  }
  os_probtraj << '{';
}

void JSONProbTrajDisplayer::endTimeTickDisplay() {
  os_probtraj << "\"tick\":" << std::setprecision(4) << std::fixed << time_tick << ",";
  if (hexfloat) {
    os_probtraj << "\"TH\":" << fmthexdouble(TH, true) << ",";
    os_probtraj << "\"ErrorTH\":"  << fmthexdouble(err_TH, true) << ",";
    os_probtraj << "\"H\":" << fmthexdouble(H, true) << ",";
  } else {
    os_probtraj << "\"TH\":" << TH << ",";
    os_probtraj << "\"ErrorTH\":" << err_TH << ",";
    os_probtraj << "\"H\":" << H << ",";
  }
  
  os_probtraj << "\"HD\":[";
  for (unsigned int nn = 0; nn <= refnode_count; nn++) {
    if (hexfloat) {
      os_probtraj << fmthexdouble(HD_v[nn], true);
    } else {
      os_probtraj << HD_v[nn];
    }
    if (nn != refnode_count) {
      os_probtraj << ",";
    }
  }
  os_probtraj << "],";

  os_probtraj << "\"probas\":[";
  unsigned int idx = 0;
  for (const Proba &proba : proba_v) {
    NetworkState network_state(proba.state, 1);
    os_probtraj << "{\"state\":\"";
    network_state.displayOneLine(os_probtraj, network);
    os_probtraj << "\",";
    if (hexfloat) {
      os_probtraj << "\"proba\":" << fmthexdouble(proba.proba, true) << ",";
      os_probtraj << "\"err_proba\":" << fmthexdouble(proba.err_proba, true);
    } else {
      os_probtraj << "\"proba\":" << std::setprecision(6) << proba.proba << ",";
      os_probtraj << "\"err_proba\":" << proba.err_proba;
    }
    os_probtraj << "}";
    if (idx < proba_v.size()-1) {
      os_probtraj << ",";
    }
    idx++;
  }
  os_probtraj << "]";
  os_probtraj << '}';
}

void JSONProbTrajDisplayer::endDisplay() {
  os_probtraj << ']';
}

void CSVProbTrajDisplayer::beginDisplay() {

  os_probtraj << "Time\tTH" << (compute_errors ? "\tErrorTH" : "") << "\tH";

  for (unsigned int jj = 0; jj <= refnode_count; ++jj) {
    os_probtraj << "\tHD=" << jj;
  }

  for (unsigned int nn = 0; nn < maxcols; ++nn) {
    os_probtraj << "\tState\tProba" << (compute_errors ? "\tErrorProba" : "");
  }

  os_probtraj << '\n';
}

void CSVProbTrajDisplayer::beginTimeTickDisplay() {
}

void CSVProbTrajDisplayer::endTimeTickDisplay() {
  os_probtraj << std::setprecision(4) << std::fixed << time_tick;
#ifdef HAS_STD_HEXFLOAT
  if (hexfloat) {
    os_probtraj << std::hexfloat;
  }
#endif
  if (hexfloat) {
    os_probtraj << '\t' << fmthexdouble(TH);
    os_probtraj << '\t' << fmthexdouble(err_TH);
    os_probtraj << '\t' << fmthexdouble(H);
  } else {
    os_probtraj << '\t' << TH;
    os_probtraj << '\t' << err_TH;
    os_probtraj << '\t' << H;
  }

  for (unsigned int nn = 0; nn <= refnode_count; nn++) {
    os_probtraj << '\t';
    if (hexfloat) {
      os_probtraj << fmthexdouble(HD_v[nn]);
    } else {
      os_probtraj << HD_v[nn];
    }
  }

  for (const Proba &proba : proba_v) {
    os_probtraj << '\t';
    NetworkState network_state(proba.state, 1);
    network_state.displayOneLine(os_probtraj, network);
    if (hexfloat) {
      os_probtraj << '\t' << fmthexdouble(proba.proba);
      os_probtraj << '\t' << fmthexdouble(proba.err_proba);
    } else {
      os_probtraj << '\t' << std::setprecision(6) << proba.proba;
      os_probtraj << '\t' << proba.err_proba;
    }
  }
  os_probtraj << '\n';
}

void CSVProbTrajDisplayer::endDisplay() {
}

