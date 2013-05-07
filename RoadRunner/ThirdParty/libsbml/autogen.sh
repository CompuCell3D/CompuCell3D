#!/bin/sh
#
# autogen.sh
#
#  Please run this file when creating or updating a configure script.
#

set -e
set -x

#
# Creates aclocal.m4 
#
# (required files)
#   1. acinclude.m4
#   2. configure.ac
#   3. config/*.m4 files (included in acinclude.m4)
#
aclocal -I config

#
# Creates a configure from configure.ac and aclocal.m4
#
# (required files)
#
#  1. aclocal.m4
#  2. configure.ac
#
autoconf
