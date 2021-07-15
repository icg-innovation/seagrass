#!/bin/bash

WORKDIR=${1:-$PWD}
DOCSDIR=$WORKDIR/docs
SOURCEDIR=$DOCSDIR/source
APPDIR=$WORKDIR/seagrass

# Modules/scripts to be excluded from documentation.
EXCLUDE=""
# Modules/scripts currently not working with Sphinx Autodoc. These exculsions can be removed once fixed.
TEMPEXCLUDE=""

# Generate all docfiles and build html documentation
(cd $DOCSDIR && sphinx-apidoc -e --force -o $SOURCEDIR $APPDIR $EXCLUDE $TEMPEXCLUDE && make html)
