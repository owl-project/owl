#!/bin/bash
allFiles=`find samples/ll/ -name '*cu' -or -name '*.h' -or -name '*cpp'`
for f in $allFiles; do
    echo $f
    cat $f | sed "s/ll->launch(0,fbSize/lloLaunch2D(llo,0,fbSize.x,fbSize.y/g" > tmp.snr
    cp tmp.snr $f
    rm tmp.snr
done
