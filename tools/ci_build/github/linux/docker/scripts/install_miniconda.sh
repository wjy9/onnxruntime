#!/bin/bash
set -e -x

MINICONDA_PREFIX=/usr/local/miniconda3
wget --no-check-certificate --no-verbose https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
chmod a+x miniconda.sh
bash miniconda.sh -b -u -p $MINICONDA_PREFIX
rm -rf miniconda.sh
$MINICONDA_PREFIX/bin/conda clean --yes --all
