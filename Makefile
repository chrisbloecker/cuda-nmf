PROG_TARGET  = nmf
C_SRCS       = matrix.c main.c config.c
CUDA_SRCS    = nmf.cu
OBJECTS      = $(C_SRCS:.c=.o) $(CUDA_SRCS:.cu=.o)
INCLUDES     = -I/usr/local/cuda/include/

CC           = g++
LD           = nvcc
NVCC         = nvcc

CCFLAGS      = -O3
NVCCFLAGS    = -O3 -D_FORCE_INLINES

compile:
	$(CC) -c $(C_SRCS) $(INCLUDES) $(CCFLAGS)
	$(NVCC) -c $(CUDA_SRCS) $(NVCCFLAGS)
	$(LD) -o $(PROG_TARGET) $(OBJECTS) -lcuda -lcublas -lcurand -lpthread

debug: CCFLAGS   += -DDEBUG
debug: NVCCFLAGS += -DDEBUG
debug: compile

all:
	@make compile
	@make doc

doc:
	doxygen

clean:
	rm -f *.o
	rm -f nmf
	rm -rf doc
	rm -f doxygen.log

.PHONY: compile debug all clean doc
