if(GZ_GPU_OUSTER_USE_CUDA)
  add_library(gz_gpu_ouster_lidar_cuda STATIC
    ray_processor_cuda.cu
  )

  set_target_properties(gz_gpu_ouster_lidar_cuda PROPERTIES
    # Compile for every architecture the installed CUDA toolkit supports.
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
