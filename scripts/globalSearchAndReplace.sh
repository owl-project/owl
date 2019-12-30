#!/bin/bash
allFiles=`find samples/ -name '*cu' -or -name '*.h' -or -name '*cpp'`
for f in $allFiles; do
    echo $f
    cat $f | sed "s/GDT/OWL/g" > tmp.snr
    cp tmp.snr $f
    rm tmp.snr
done
