#!/bin/bash
allFiles=`find . -name '*cu' -or -name '*.h' -or -name '*cpp'`
for f in $allFiles; do
    echo $f
    cat $f | sed "s/createUserGeomGroup/userGeomGroupCreate/g" > $f
done
