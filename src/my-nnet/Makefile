#KALDISRC=/home/loach/kaldi/kaldi-trunk/src
include ../../path

include ../config

#all: CXXFLAGS += -std=c++11
all: $(LIBFILE)
include $(KALDISRC)/kaldi.mk

CXXFLAGS += -I$(KALDISRC) -I..
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS) -lpthread

BINFILES = 

OBJFILES = nnet-my-nnet.o nnet-loss-strt.o nnet-my-loss.o nnet-my-component.o  nnet-my-rennet.o

TESTFILES = 

ADDLIBS = 

LIBNAME = my-nnet

include $(KALDISRC)/makefiles/default_rules.mk

