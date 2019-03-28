include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = hotspot.cu

EXE = hotspot

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)  -lcudart --cudart=shared


enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)  -lcudart --cudart=shared


debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)  -lcudart --cudart=shared


debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR)  -lcudart --cudart=shared


clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
