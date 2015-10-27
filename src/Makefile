#KALDISRC=/home/loach/kaldi/kaldi-trunk/src
include ../path

O3: CXXFLAGS += -O3 -fopenmp
O3: NVCCFLAGS += -O3
O3: all

#all: CXXFLAGS += -std=c++11
all: $(LIBFILE)
EXTRA_CXXFLAGS = -Wno-sign-compare
include $(KALDISRC)/kaldi.mk

CXXFLAGS += -I$(KALDISRC)
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS) -lpthread

BINFILES = post-phn acc con-svm nnet-pop lab-lda gen-lab lattice-to-vec \
           path-fer split-score-path\
           trim-path svmout-to-vec combine-score-path init-score-path\
	   score-baseline lattice-baseline post-to-vec vec-to-score-path\
	   score-path-copy weight-basefloat best-score-path weight-score-path\
	   normalize-score-path exp-score-path log-score-path cmvn-score-path replace-feats \
	   score-oracle score-path-point snnet-score int32-to-uchar uchar-to-int32 \
	   snnet-train-pairshuff snnet-psi-cmvn snnet-gen-psi snnet-train-listshuff\
	   srnnet2-train-listshuff srnnet2-score srnnet2-init srnnet2-decode srnnet2-train-frameshuff\
	   naiive msnnet-init msnnet-score msnnet-train-listshuff

OBJFILES = kernel.o util.o strt.o cumat.o snnet.o bnnet.o srnnet2.o nnet-mux-component.o msnnet.o

TESTFILES = 

ADDLIBS = $(KALDISRC)/feat/kaldi-feat.a $(KALDISRC)/transform/kaldi-transform.a \
          $(KALDISRC)/gmm/kaldi-gmm.a $(KALDISRC)/thread/kaldi-thread.a \
          $(KALDISRC)/nnet/kaldi-nnet.a $(KALDISRC)/cudamatrix/kaldi-cudamatrix.a \
          $(KALDISRC)/lat/kaldi-lat.a $(KALDISRC)/hmm/kaldi-hmm.a \
          $(KALDISRC)/tree/kaldi-tree.a $(KALDISRC)/matrix/kaldi-matrix.a \
          $(KALDISRC)/util/kaldi-util.a $(KALDISRC)/base/kaldi-base.a \
          mycudakernel.a

LIBNAME = mycudakernel

include $(KALDISRC)/makefiles/default_rules.mk

ifeq ($(CUDA), true)
  #Default compute capability architectures we compile with
  CUDA_ARCH=-gencode arch=compute_20,code=sm_20
  #Get the CUDA Toolkit version (remove decimal point char)
  CUDA_VERSION=$(shell $(CUDATKDIR)/bin/nvcc -V | grep release | sed -e 's|.*release ||' -e 's|,.*||' -e 's|\.||')
  #For toolkit 4.2 or newer, add the compute capability 3.0 
  CUDA_VER_GT_4_2 := $(shell [ $(CUDA_VERSION) -ge 42 ] && echo true)
  ifeq ($(CUDA_VER_GT_4_2), true)
    CUDA_ARCH += -gencode arch=compute_30,code=sm_30
  endif
  #For toolkit 5.0 or newer, add the compute capability 3.5 
  CUDA_VER_GT_5_0 := $(shell [ $(CUDA_VERSION) -ge 50 ] && echo true)
  ifeq ($(CUDA_VER_GT_5_0), true)
    CUDA_ARCH += -gencode arch=compute_35,code=sm_35
  endif
  #For toolkit 6.0 or newer, add the compute capability 5.0
  CUDA_VER_GT_6_0 := $(shell [ $(CUDA_VERSION) -ge 60 ] && echo true)
  ifeq ($(CUDA_VER_GT_6_0), true)
    CUDA_ARCH += -gencode arch=compute_50,code=sm_50
  endif
  #For toolkit older than 6.5, add the compute capability 1.0
  CUDA_VER_GT_6_5 := $(shell [ $(CUDA_VERSION) -ge 65 ] && echo true)
  ifneq ($(CUDA_VER_GT_6_5), true)
    CUDA_ARCH += -gencode arch=compute_13,code=sm_13 \
                 -gencode arch=compute_10,code=sm_10 
  endif
endif

%.o : %.cu 
	$(CUDATKDIR)/bin/nvcc -g -G $(NVCCFLAGS) -c $< -o $@ $(CUDA_INCLUDE) $(CUDA_FLAGS) $(CUDA_ARCH) -I$(KALDISRC)
