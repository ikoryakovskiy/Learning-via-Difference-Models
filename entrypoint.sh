#!/bin/bash
set -e

file="/grl/qt-build/py_env.*"
if [ -f $file ]; then
  ln -sf $file /drl
else
  echo "$file not found. Please, rebuild GRL."
fi

exec "$@"
