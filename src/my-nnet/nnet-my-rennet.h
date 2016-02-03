#ifndef _MYRENNET_H_
#define _MYRENNET_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "my-utils/util.h"
#include "my-nnet/nnet-my-component.h"
#include "my-nnet/nnet-my-nnet.h"
#include "my-nnet/nnet-att-component.h"
#include <iostream>
#include <string>
#include <vector>


using namespace std;
using namespace kaldi;

class NnetBatch : public MyNnet{
   public:
      NnetBatch():batch_num_(0){}
      NnetBatch(const NnetBatch& other);
      NnetBatch &operator = (const NnetBatch &other);
      
      NnetBatch* Copy() const { return new NnetBatch(*this); }

      ~NnetBatch() { Destroy(); }

   public:
      void Propagate(int index, const CuMatrixBase<BaseFloat> &in, MyCuMatrix<BaseFloat> *out);

      // only do backpropagte no update...
      void Backpropagate(int index, const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff, MyCuMatrix<BaseFloat> *in_diff);

      const vector< MyCuMatrix<BaseFloat> >& PropagateBuffer(int index) const{
         return propagate_buf_arr_[index];
      }

      const vector< MyCuMatrix<BaseFloat> >& BackpropagateBuffer(int index) const {
         return backpropagate_buf_arr_[index];
      }

      string InfoPropagate(int index) const;

      string InfoBackPropagate(int index) const;

      void SetBatchNum(int32 batch_num);

      void Check() const;

      void Destroy();

   private:

      vector< vector< MyCuMatrix<BaseFloat> > > propagate_buf_arr_;
      vector< vector< MyCuMatrix<BaseFloat> > > backpropagate_buf_arr_;

      int32 batch_num_;
};

class RENnet{
   public:
      RENnet():sel_model_(NULL), nnet_in_(NULL), nnet_out_(NULL), att_model_(NULL), depth_(0){}
      RENnet(const RENnet& other);
      RENnet &operator = (const RENnet &other);

      ~RENnet() { Destroy(); }

   public:
      void Propagate(const CuMatrixBase<BaseFloat> &in, MyCuMatrix<BaseFloat> *out);

      // only do backpropagte no update...
      void Backpropagate(const CuMatrixBase<BaseFloat> &out_diff, MyCuMatrix<BaseFloat> *in_diff);
      // update accumulated gradient.
      void Update();

      void Feedforward(const CuMatrixBase<BaseFloat> &in, MyCuMatrix<BaseFloat> *out);

      int32 InputDim() const;
      int32 OutputDim() const;

      int32 NumSubModels() const { return sub_models_.size(); }

      const NnetBatch& GetSubModel(int32 i) const;
      NnetBatch& GetSubModel(int32 i);

      void SetSubModel(int32 i, NnetBatch *nnet_batch);

      void SetSelModel(NnetBatch *nnet_batch);

      int32 NumParams() const;

      void GetParams(Vector<BaseFloat> *weight) const;

      // for Dropout
      void SetDropoutRetention(BaseFloat r);

      void Init(istream &is);
      void Init(const string& config_file);
      
      void Read(const string& file);

      void Read(istream &is, bool binary);

      void Write(const string &file, bool binary) const;
      
      void Write(ostream &os, bool binary) const;

      string Info() const;

      string InfoGradient() const;

      string InfoPropagate() const;

      string InfoBackPropagate() const;

      void SetDepth(int32 depth);

      void Check() const;

      void Destroy();

      void SetTrainOptions(const NnetTrainOptions& opts);
      
      const NnetTrainOptions& GetTrainOptions() const{
         return opts_;
      }

      void SetNnetIn(MyNnet* nnet){ nnet_in_ = nnet; }

      void SetNnetOut(MyNnet* nnet) { nnet_out_ = nnet; }

   private:
      void CheckBuff() const;

      vector<NnetBatch*> sub_models_;
      NnetBatch*         sel_model_;
      MyNnet*            nnet_in_;
      MyNnet*            nnet_out_;
      AttComponent*      att_model_;

      int32              depth_;

      NnetTrainOptions   opts_;

      MyCuMatrix<BaseFloat>           tmp_cumat_;
      vector< MyCuMatrix<BaseFloat> > propagate_buf_;
      vector< MyCuMatrix<BaseFloat> > propagate_buf_att_;
      vector< vector< MyCuMatrix<BaseFloat> > > propagate_buf_sub_;

      vector< MyCuMatrix<BaseFloat> > backpropagate_buf_;
      vector< MyCuMatrix<BaseFloat> > backpropagate_buf_att_;
      vector< vector< MyCuMatrix<BaseFloat> > > backpropagate_buf_sub_;
};

#endif
