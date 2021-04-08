#!/bin/bash
set -e -x

while getopts f:n parameter_Option
do case "${parameter_Option}"
in
f) REQUIREMENTS=${OPTARG};;
n) ENV_NAME=${OPTARG};;
esac
done

MINICONDA_PREFIX=/usr/local/miniconda3

$MINICONDA_PREFIX/envs/$ENV_NAME/bin/pip install --force-reinstall -r $REQUIREMENTS