#!/bin/bash
set -e

file="/grl/qt-build/py_env.*"
if [ -f $file ]; then
  cp $file /openai
else
  echo "$file not found. Please, rebuild GRL."
fi

exec "$@"
