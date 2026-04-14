if(GZ_GPU_OUSTER_USE_CUDA)
  add_library(gz_gpu_ouster_lidar_cuda STATIC
    ray_processor_cuda.cu
  )

  set_target_properties(gz_gpu_ouster_lidar_cuda PROPERTIES
    # Compile for every major architecture the installed CUDA toolkit supports.
    # "all-major" typically covers sm_50+ (Maxwell) through the newest arch.
    # Minimum required: sm_50 (Maxwell).  Older Kepler (sm_3x) GPUs are NOT
    # supported — they lack the __shfl_sync and cooperative-group intrinsics
    # used by the ray-processing kernels.  If your GPU is older than Maxwell,
    # the plugin falls back to the CPU implementation automatically (see the
    # else branch below).
    CUDA_ARCHITECTURES all-major
    POSITION_INDEPENDENT_CODE ON
  )
else()
  add_library(gz_gpu_ouster_lidar_cuda STATIC
    ray_processor_fallback.cpp
  )
  set_target_properties(gz_gpu_ouster_lidar_cuda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
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
