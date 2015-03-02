KALDISRC=/home/loach/kaldi/kaldi-trunk/src

all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include $(KALDISRC)/kaldi.mk

CXXFLAGS += -I$(KALDISRC)
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS) -lpthread

BINFILES = post-phn acc con-svm nnet-pop lab-lda gen-lab lattice-to-vec \
           snnet-train-shuff snnet-test snnet-gibbs path-fer snnet-gibbs2 split-path-score\
           trim-path svmout-to-vec combine-score-path

OBJFILES = 

TESTFILES =

ADDLIBS = $(KALDISRC)/feat/kaldi-feat.a $(KALDISRC)/transform/kaldi-transform.a \
          $(KALDISRC)/gmm/kaldi-gmm.a $(KALDISRC)/thread/kaldi-thread.a \
          $(KALDISRC)/nnet/kaldi-nnet.a $(KALDISRC)/cudamatrix/kaldi-cudamatrix.a \
          $(KALDISRC)/lat/kaldi-lat.a $(KALDISRC)/hmm/kaldi-hmm.a \
          $(KALDISRC)/tree/kaldi-tree.a $(KALDISRC)/matrix/kaldi-matrix.a \
          $(KALDISRC)/util/kaldi-util.a $(KALDISRC)/base/kaldi-base.a 

include $(KALDISRC)/makefiles/default_rules.mk

