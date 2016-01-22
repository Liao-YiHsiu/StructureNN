#KALDISRC=/home/loach/kaldi/kaldi-trunk/src
include ../../path

O3: CXXFLAGS += -O3 -fopenmp
O3: NVCCFLAGS += -O3
O3: all

#all: CXXFLAGS += -std=c++11
all: $(LIBFILE)
EXTRA_CXXFLAGS = -Wno-sign-compare
include $(KALDISRC)/kaldi.mk

CXXFLAGS += -I$(KALDISRC) -I..
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS) -lpthread

BINFILES = mynnet-info mynnet-init mynnet-score mynnet-train-single-utt mynnet-train-utt 

OBJFILES = 

TESTFILES = 

ADDLIBS = ../my-nnet/my-nnet.a ../my-cumatrix/my-cu-matrix.a \
	  ../score-path/score-path.a ../my-cuda-kernel/my-cuda-kernel.a \
	  ../my-utils/my-util.a \
	  $(KALDISRC)/feat/kaldi-feat.a $(KALDISRC)/transform/kaldi-transform.a \
	  $(KALDISRC)/gmm/kaldi-gmm.a $(KALDISRC)/thread/kaldi-thread.a \
	  $(KALDISRC)/nnet/kaldi-nnet.a $(KALDISRC)/cudamatrix/kaldi-cudamatrix.a \
	  $(KALDISRC)/lat/kaldi-lat.a $(KALDISRC)/hmm/kaldi-hmm.a \
	  $(KALDISRC)/tree/kaldi-tree.a $(KALDISRC)/matrix/kaldi-matrix.a \
	  $(KALDISRC)/util/kaldi-util.a $(KALDISRC)/base/kaldi-base.a \

LIBNAME = 

include $(KALDISRC)/makefiles/default_rules.mk

