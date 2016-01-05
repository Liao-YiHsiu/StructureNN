#!/bin/bash

dir=$(ls -t data | head -n 1)
model=$(ls -t data/$dir | head -n 1)

./utils/plot.sh data/$dir/$model/log
