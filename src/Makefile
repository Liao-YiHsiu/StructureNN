.PHONY: clean

#LDFLAGS+=-lntl -lm -lgmp # link to math library
LDLIBS+=  
CXXFLAGS+= 
KALDISRC=/home/loach/kaldi/kaldi-trunk/src

tmp:
	g++ -msse -msse2 -Wall -I$(KALDISRC) -pthread -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN -Wno-sign-compare -Wno-unused-local-typedefs -Winit-self -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H -DHAVE_ATLAS -I/home/loach/kaldi/kaldi-trunk/tools/ATLAS/include -I/home/loach/kaldi/kaldi-trunk/tools/openfst/include -Wno-sign-compare -g  -DHAVE_CUDA -I/usr/local/cuda/include    -c -o str_format.o str_format.cpp
	g++ -rdynamic -Wl,-rpath=/home/loach/kaldi/kaldi-trunk/tools/openfst/lib  str_format.o $(KALDISRC)/feat/kaldi-feat.a $(KALDISRC)/transform/kaldi-transform.a $(KALDISRC)/gmm/kaldi-gmm.a $(KALDISRC)/thread/kaldi-thread.a $(KALDISRC)/tree/kaldi-tree.a $(KALDISRC)/matrix/kaldi-matrix.a $(KALDISRC)/util/kaldi-util.a $(KALDISRC)/base/kaldi-base.a   -L/home/loach/kaldi/kaldi-trunk/tools/openfst/lib -lfst /usr/lib/liblapack_atlas.so /usr/lib/libcblas.so /usr/lib/libatlas.so /usr/lib/libf77blas.so -lm -lpthread -ldl -o str_format
	g++ -msse -msse2 -Wall -I$(KALDISRC) -pthread -DKALDI_DOUBLEPRECISION=0 -DHAVE_POSIX_MEMALIGN -Wno-sign-compare -Wno-unused-local-typedefs -Winit-self -DHAVE_EXECINFO_H=1 -rdynamic -DHAVE_CXXABI_H -DHAVE_ATLAS -I/home/loach/kaldi/kaldi-trunk/tools/ATLAS/include -I/home/loach/kaldi/kaldi-trunk/tools/openfst/include -Wno-sign-compare -g  -DHAVE_CUDA -I/usr/local/cuda/include    -c -o svm_hmm.o svm_hmm.cpp
	g++ -rdynamic -Wl,-rpath=/home/loach/kaldi/kaldi-trunk/tools/openfst/lib  svm_hmm.o $(KALDISRC)/feat/kaldi-feat.a $(KALDISRC)/transform/kaldi-transform.a $(KALDISRC)/gmm/kaldi-gmm.a $(KALDISRC)/thread/kaldi-thread.a $(KALDISRC)/tree/kaldi-tree.a $(KALDISRC)/matrix/kaldi-matrix.a $(KALDISRC)/util/kaldi-util.a $(KALDISRC)/base/kaldi-base.a   -L/home/loach/kaldi/kaldi-trunk/tools/openfst/lib -lfst /usr/lib/liblapack_atlas.so /usr/lib/libcblas.so /usr/lib/libatlas.so /usr/lib/libf77blas.so -lm -lpthread -ldl -o svm_hmm 

TARGET= str_format htk_lat2path 

all: $(TARGET)
# type make/make all to compile test_hmm

o3: CPPFLAGS+=-o3
o3: all

debug: CPPFLAGS+=-g
debug: all

run:
	./nn_output.sh
svm:
	./svm_hmm.sh
	../svm_hmm/svm_hmm_learn -c 5 -e 0.5 out svm.model

clean:
	$(RM) $(TARGET)   # type make clean to remove the compiled file
