# Streaming kernels of the large-dataset passes (experiments/bigsets): per-query top-k filtering over score blocks.
#   make -C csrc        -> csrc/libbigkern.so (loaded by pcafair/bigkern.py)
all: libbigkern.so
libbigkern.so: bigkern.cpp
	g++ -O3 -march=native -ffast-math -fopenmp -shared -fPIC bigkern.cpp -o libbigkern.so
clean:
	rm -f libbigkern.so
