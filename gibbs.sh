#!/bin/bash

sample_methods="weight;max;ratio"
starts=";ark:lattice-to-nbest --n=1 ark:data/nn_post/test.lat ark:- |lattice-to-vec ark:- ark:- |; ark:data/nn_post/test.lab"

IFS=';'
i=0
for sample_method in $sample_methods
do
   for start in $starts
   do
      snnet-gibbs2 --start=$start --sample-method=$sample_method ark:data/nn_post/test.ark data/nn_post/data_nn.model ark,t:tmp${i}.ark
      i=$(( i + 1 ))
   done
done

