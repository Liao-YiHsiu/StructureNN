SUBDIRS = my-cuda-kernel my-cumatrix my-nnet score-path my-utils my-nnetbin

all: $(SUBDIRS)

.PHONY: $(SUBDIRS)

$(SUBDIRS) :
	$(MAKE) -C $@ 

clean:
	-for x in $(SUBDIRS); do $(MAKE) -C $$x clean; done

#library denpendency
my-cuda-kernel:
my-cumatrix: my-cuda-kernel
score-path:
my-utils: score-path
my-nnet: my-cuda-kernel my-cumatrix my-utils
my-nnetbin: my-nnet score-path my-cumatrix
