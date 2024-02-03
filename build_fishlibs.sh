nvcc plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu -std=c++11 -O3 --restrict --ptx --gpu-architecture=compute_86 --gpu-code=sm_86 -o plugins/cuda/resources/kaspa-cuda-sm86.ptx -Xptxas -O3 -Xcompiler -O3

nvcc plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu -std=c++11 -O3 --restrict --ptx --gpu-architecture=compute_75 --gpu-code=sm_75 -o plugins/cuda/resources/kaspa-cuda-sm75.ptx -Xptxas -O3 -Xcompiler -O3

nvcc plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu -std=c++11 -O3 --restrict --ptx --gpu-architecture=compute_61 --gpu-code=sm_61 -o plugins/cuda/resources/kaspa-cuda-sm61.ptx -Xptxas -O3 -Xcompiler -O3

nvcc plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu -ccbin=gcc-7 -std=c++11 -O3 --restrict --ptx --gpu-architecture=compute_30 --gpu-code=sm_30 -o plugins/cuda/resources/kaspa-cuda-sm30.ptx

nvcc plugins/cuda/kaspa-cuda-native/src/kaspa-cuda.cu -ccbin=gcc-5 -std=c++11 -O3 --restrict --ptx --gpu-architecture=compute_20 --gpu-code=sm_20 -o plugins/cuda/resources/kaspa-cuda-sm20.ptx

cargo build --release



