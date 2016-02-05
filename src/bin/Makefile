
all:
	find .. -type f -executable -print | \
	   while read file; do \
	   mv $$file . ;\
	   done
clean:
	find . -type f -executable -print | \
	   while read file; do \
	   rm $$file ;\
	   done
