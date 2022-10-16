#!/bin/bash

x=1
while [ $x -le 200 ]
do
  echo $x
  python Synthetic_end-to-end.py $x
  x=$(( $x + 1 ))
done
