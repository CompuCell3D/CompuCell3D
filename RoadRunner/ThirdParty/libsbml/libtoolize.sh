#!/bin/sh
#
# libtoolize.sh
#
# Please run this script when importing or updating libtool resources 
# for libSBML, and then run "make configure" to update configure.
#
# This script will create or update the following files:
#
#   config/ltmain.sh
#   config/config.guess
#   config/config.sub
#
# (*) config.guess and config.sub should be manually updated to the 
#     latest versions which are available at ftp://ftp.gnu.org/gnu/config
#

LIBTOOLIZE=libtoolize

set -e
set -x

#
# Creates config/{ltmain.sh, config.guess, config.sub}
#
if ${LIBTOOLIZE} --version | grep 'libtoolize (GNU libtool) 2'; then
  #
  # In libtool 2.2.x, "-i" optsion is required to copy missing 
  # auxiliary files.
  #
  ${LIBTOOLIZE} -c -f -i
else
  ${LIBTOOLIZE} -c -f
fi

set +x
echo "Please run ./autogen.sh to update configure script."

