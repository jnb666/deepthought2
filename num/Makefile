all: libkernels.a

libkernels.a: kernels.o
	ar cru libkernels.a kernels.o
	ranlib libkernels.a

kernels.o: kernels.cu cuda.h
	nvcc -m64 -arch sm_61 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -c -o kernels.o kernels.cu

clean:
	rm *.o