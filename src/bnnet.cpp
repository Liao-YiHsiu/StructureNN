#include "bnnet.h"

void BNnet::Propagate(const vector< CuMatrix<BaseFloat> > &in_arr, 
      vector< CuMatrix<BaseFloat> > &out_arr, int N){

   CuMatrix<BaseFloat> forward;
   CuMatrix<BaseFloat> out;

   VecToMat(in_arr, forward, N);

   Nnet::Propagate(forward, &out);

   MatToVec(out, in_arr, out_arr, N);
}

void BNnet::Backpropagate(const vector< CuMatrix<BaseFloat> > &out_diff,
      vector< CuMatrix<BaseFloat> > *in_diff){

   if(NumComponents() == 1 && GetComponent(0).GetType() == Component::kSplice){
      if(in_diff != NULL){
         if(in_diff->size() != out_diff.size())
            in_diff->resize(out_diff.size());
         for(int i = 0; i < out_diff.size(); ++i)
            (*in_diff)[i] = out_diff[i];
      }
      return;
   }

   CuMatrix<BaseFloat> backward;
   VecToMat(out_diff, backward);

   if( in_diff == NULL){
      Nnet::Backpropagate(backward, NULL);
   } else {
      CuMatrix<BaseFloat> backward_in_diff;
      Nnet::Backpropagate(backward, &backward_in_diff);
      MatToVec(backward_in_diff, out_diff, *in_diff);
   }
}

// if the out_diff for all input are the same.
void BNnet::Backpropagate(const CuMatrix<BaseFloat> &out_diff,
      vector< CuMatrix<BaseFloat> > *in_diff, int N){
   if(in_diff != NULL) assert(N > 0);

   if(NumComponents() == 1 && GetComponent(0).GetType() == Component::kSplice){
      if(in_diff != NULL){
         if(in_diff->size() < N)
            in_diff->resize(N);
         for(int i = 0; i < N; ++i)
            (*in_diff)[i] = out_diff;
      }
      return;
   }

   CuMatrix<BaseFloat> backward;
   RepMat(out_diff, backward, N);

   if( in_diff == NULL){
      Nnet::Backpropagate(backward, NULL);
   } else {
      CuMatrix<BaseFloat> backward_in_diff;
      Nnet::Backpropagate(backward, &backward_in_diff);

      vector<int> ref(N);
      for(int i = 0; i < N; ++i)
         ref[i] = out_diff.NumRows();

      MatToVec(backward_in_diff, ref, *in_diff);
   }
}

void BNnet::Feedforward(const vector< CuMatrix<BaseFloat> > &in_arr,
      vector< CuMatrix<BaseFloat> > &out_arr){

   CuMatrix<BaseFloat> forward;
   CuMatrix<BaseFloat> out;

   VecToMat(in_arr, forward);

   Nnet::Feedforward(forward, &out);

   MatToVec(out, in_arr, out_arr);
}

void BNnet::Propagate(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out) {
   Nnet::Propagate(in, out);
}

void BNnet::Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, CuMatrix<BaseFloat> *in_diff){
   if(NumComponents() == 1 && GetComponent(0).GetType() == Component::kSplice){
      if(in_diff != NULL)
         *in_diff = out_diff;
      return;
   }
   Nnet::Backpropagate(out_diff, in_diff);
}

void BNnet::Feedforward(const CuMatrixBase<BaseFloat> &in, CuMatrix<BaseFloat> *out){
   Nnet::Feedforward(in, out);
}

