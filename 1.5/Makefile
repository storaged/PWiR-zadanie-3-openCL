CPPFLAGS = -Wall -I /usr/local/cuda-5.0/include/
LDFLAGS	= -L /usr/lib64/nvidia
LDLIBS = -lOpenCL -lstdc++

EXECS = knn

TARGET: $(EXECS)

knn.o:	knn.cpp knn.hpp

.PHONY: clean TARGET
clean:
	rm -f $(EXECS) *.o *~ 

