#KALDISRC=/home/loach/kaldi/kaldi-trunk/src
include ../../path

include ../config

#all: CXXFLAGS += -std=c++11
all: $(LIBFILE)
include $(KALDISRC)/kaldi.mk

CXXFLAGS += -I$(KALDISRC) -I..
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS) -lpthread

BINFILES = con-svm gen-lab int32-to-uchar lattice-all-vec lattice-baseline naiive path-fer\
	   post-phn post-to-vec replace-feats score-baseline score-oracle shuffle-ark shuffle-lab\
	   svmout-to-vec trim-path uchar-to-int32 weight-basefloat nnet-pop

OBJFILES = util.o

TESTFILES = 

ADDLIBS = $(KALDISRC)/hmm/kaldi-hmm.a $(KALDISRC)/util/kaldi-util.a \
	  $(KALDISRC)/nnet/kaldi-nnet.a $(KALDISRC)/cudamatrix/kaldi-cudamatrix.a \
	  $(KALDISRC)/matrix/kaldi-matrix.a $(KALDISRC)/base/kaldi-base.a  \
	  $(KALDISRC)/lat/kaldi-lat.a $(KALDISRC)/tree/kaldi-tree.a \
	  my-util.a  ../score-path/score-path.a
	  

LIBNAME = my-util

include $(KALDISRC)/makefiles/default_rules.mk

