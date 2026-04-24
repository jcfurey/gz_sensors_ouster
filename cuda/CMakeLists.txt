if(GZ_GPU_OUSTER_USE_CUDA)
  # CUDA build: CUDA path owns the class, and still links the pure-CPU
  # implementation so it can dispatch to it at runtime when no CUDA-capable
  # device is detected (e.g. container started without GPU passthrough).
  add_library(gz_gpu_ouster_lidar_cuda STATIC
    ray_processor_cuda.cu
    ray_processor_cpu_impl.cpp
  )

  set_target_properties(gz_gpu_ouster_lidar_cuda PROPERTIES
    # Compile for every major architecture the installed CUDA toolkit supports.
    # "all-major" typically covers sm_50+ (Maxwell) through the newest arch.
    # Minimum required: sm_50 (Maxwell). Older Kepler (sm_3x) GPUs are NOT
    # supported — they lack the __shfl_sync and cooperative-group intrinsics
    # used by the ray-processing kernels. On such GPUs (or when no GPU is
    # present at all) the runtime probe in CudaRayProcessor() flips to the
    # CPU implementation automatically.
    CUDA_ARCHITECTURES all-major
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )
else()
  # Non-CUDA build: no CUDA toolkit available. Class methods in
  # ray_processor_fallback.cpp delegate straight to the CPU free functions.
  add_library(gz_gpu_ouster_lidar_cuda STATIC
    ray_processor_fallback.cpp
    ray_processor_cpu_impl.cpp
  )
  set_target_properties(gz_gpu_ouster_lidar_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )
endif()

target_include_directories(gz_gpu_ouster_lidar_cuda PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(GZ_GPU_OUSTER_USE_CUDA)
  target_link_libraries(gz_gpu_ouster_lidar_cuda
    CUDA::curand
  )
endif()

# OpenMP for CPU resampling in fallback path (and processRaw in CUDA path).
if(OpenMP_CXX_FOUND)
  target_link_libraries(gz_gpu_ouster_lidar_cuda OpenMP::OpenMP_CXX)
endif()
