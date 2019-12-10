#!/bin/bash
allFiles=`find samples/ owl/ -name '*cu' -or -name '*.h' -or -name '*cpp'`
for f in $allFiles; do
    echo $f
    cat $f | sed "s/realloc/alloc/g" > tmp.snr
    cp tmp.snr $f
    rm tmp.snr
done
