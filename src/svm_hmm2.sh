#!/bin/bash
. ../timit/path.sh

   ./svm_hmm2 simp/label/train.lab ark,s,cs:simp/prob/fbank_cd414/train.prob simp/data.out  
   ../svm_hmm/svm_hmm_learn -c 1000 -e 0.5 simp/data.out simp/data.model &>simp/data.log 
   
   echo "SVM testing start..................................."
   ./svm_hmm2 simp/label/test.lab ark,s,cs:simp/prob/fbank_cd414/test.prob simp/test.out  
   ../svm_hmm/svm_hmm_classify simp/test.out simp/data.model simp/test.tags &>>simp/data.log 

