#!/usr/bin/env bash
# for i in `seq 1 40`; do
for i in `seq $1 $2`; do
    printf "Downloading %.2d out of 40\n" $i;
    partNum=$(printf "%.2d" $i)
    cmd='wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/ox100k/oxc1_100k.part'$partNum'.rar';
    echo 'Executing: '$cmd
    $cmd
done
