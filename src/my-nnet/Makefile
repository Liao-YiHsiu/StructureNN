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

BINFILES = 

OBJFILES = nnet-my-nnet.o nnet-loss-strt.o nnet-my-loss.o nnet-my-component.o 

TESTFILES = 

ADDLIBS = 

LIBNAME = my-nnet

include $(KALDISRC)/makefiles/default_rules.mk

