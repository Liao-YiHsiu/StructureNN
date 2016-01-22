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

OBJFILES = cu-matrix.o cu-general.o cu-utils.o

TESTFILES = 

ADDLIBS = 

LIBNAME = my-cu-matrix

include $(KALDISRC)/makefiles/default_rules.mk

