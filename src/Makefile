KALDISRC=/home/loach/kaldi/kaldi-trunk/src

all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include $(KALDISRC)/kaldi.mk
CXXFLAGS += -I$(KALDISRC)

BINFILES = svm_hmm post2phn acc combine svm_hmm2

OBJFILES = 

TESTFILES =

ADDLIBS = $(KALDISRC)/feat/kaldi-feat.a $(KALDISRC)/transform/kaldi-transform.a \
          $(KALDISRC)/gmm/kaldi-gmm.a $(KALDISRC)/thread/kaldi-thread.a \
          $(KALDISRC)/tree/kaldi-tree.a $(KALDISRC)/matrix/kaldi-matrix.a \
          $(KALDISRC)/util/kaldi-util.a $(KALDISRC)/base/kaldi-base.a

include $(KALDISRC)/makefiles/default_rules.mk

run: $(BINFILES)
	./nn_output.sh
svm: $(BINFILES)
	./svm_hmm.sh
svm2: $(BINFILES)
	./svm_hmm2.sh
