#KALDISRC=/home/loach/kaldi/kaldi-trunk/src
include ../../path

include ../config

#all: CXXFLAGS += -std=c++11
all: $(LIBFILE)
include $(KALDISRC)/kaldi.mk

CXXFLAGS += -I$(KALDISRC) -I..
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS) -lpthread

BINFILES = best-score-path cmvn-score-path combine-score-path exp-score-path init-score-path \
	   log-score-path normalize-score-path score-path-copy score-path-point shuffle-score-path \
	   split-score-path vec-to-score-path weight-score-path

OBJFILES = score-path.o

TESTFILES = 

ADDLIBS = $(KALDISRC)/util/kaldi-util.a $(KALDISRC)/matrix/kaldi-matrix.a $(KALDISRC)/base/kaldi-base.a  \
	  score-path.a

LIBNAME = score-path

include $(KALDISRC)/makefiles/default_rules.mk

