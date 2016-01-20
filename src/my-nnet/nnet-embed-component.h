#ifndef EMBED_COMPONENT_H_
#define EMBED_COMPONENT_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "util/text-utils.h"
#include "util.h"

#include "nnet-my-component.h"

#include <algorithm>
#include <sstream>

using namespace std;

class Embed : public MyComponent{
   public:
      Embed(int32 input_dim, int32 output_dim):
         MyComponent(input_dim, output_dim), label_num_(1), seq_stride_(1){}
      virtual ~Embed() {}

      bool IsEmbed() const{ return true;}

      void Propagate(const CuMatrixBase<BaseFloat> &in, 
            CuMatrix<BaseFloat> *out);

      void Backpropagate(const CuMatrixBase<BaseFloat> &in,
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            CuMatrix<BaseFloat> *in_diff);

      virtual void InitData(istream &is);
      virtual void ReadData(istream &is, bool binary);
      virtual void WriteData(ostream &os, bool binary) const;

      // embedding label seqs...
      // seq is 0-based
      // each input row could have multiple seqs.
      void SetLabelSeqs(const vector<int32> &seq, int seq_stride);
      void SetLabelNum(int label_num);
      int GetLabelNum() const { return label_num_; }

   protected:
      virtual void SetLabelSeqsFnc(const vector<int32> &seq, int seq_stride) {}
      virtual void SetLabelNumFnc(int label_num) {}

   protected:
      int32 label_num_;
      int32 seq_stride_;
      vector<int32> seq_;
};

class EmbedSimple : public Embed{
   public:
      EmbedSimple(int32 input_dim, int32 output_dim):
         Embed(input_dim, output_dim){}
      ~EmbedSimple() {}
      
      Component* Copy() const;

      //string Info() const;
      //string InfoGradient() const;
      NOT_UPDATABLE();


      MyType myGetType() const { return mEmbedSimple; }

   protected:
      void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
            CuMatrixBase<BaseFloat> *out);

      void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, 
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            CuMatrixBase<BaseFloat> *in_diff);

      void SetLabelSeqsFnc(const vector<int32> &seq, int seq_stride);
      void SetLabelNumFnc(int label_num);

   private:

      CuVectorG< int32 > seq_device_;
};

class EmbedMux : public Embed {
   public:
      EmbedMux(int32 input_dim, int32 output_dim):
         Embed(input_dim, output_dim),
         comps_(0), in_buff_(0), in_diff_buff_(0),
         out_buff_(0), out_diff_buff_(0),
         cnt_(0) {}
      ~EmbedMux();
      
      Component* Copy() const;

      string Info() const;
      string InfoGradient() const;

      MyType myGetType() const { return mEmbedMux; }

      // updatable componets
      bool IsUpdatable() const { return true; }
      int32 NumParams() const;
      void GetParams(Vector<BaseFloat> *params) const;
      void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff);

      void SetTrainOptions(const NnetTrainOptions &opts);

   protected:
      void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
            CuMatrixBase<BaseFloat> *out);

      void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, 
            const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff,
            CuMatrixBase<BaseFloat> *in_diff);

      void InitData(istream &is);
      void ReadData(istream &is, bool binary);
      void WriteData(ostream &os, bool binary) const;

      void SetLabelSeqsFnc(const vector<int32> &seq, int seq_stride);
      void SetLabelNumFnc(int label_num);

   private:
      BaseFloat** getVecCuMatrixPt(vector< CuMatrix<BaseFloat> > &mat_arr);
      int32*      getVecCuMatrixStride(vector< CuMatrix<BaseFloat> > &mat_arr);

      vector< Component* > comps_;
      vector< CuMatrix<BaseFloat> > in_buff_; 
      vector< CuMatrix<BaseFloat> > in_diff_buff_; 
      vector< CuMatrix<BaseFloat> > out_buff_;
      vector< CuMatrix<BaseFloat> > out_diff_buff_;

      vector<int32> id_;
      vector<int32> cnt_;

      CuVectorG<int32> seq_device_;
      CuVectorG<int32> id_device_;

      CuVectorG< int32 >      mat_arr_stride_device_;
      CuVectorG< BaseFloat* > mat_arr_pt_device_;
};
#endif
