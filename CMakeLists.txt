cmake_minimum_required (VERSION 2.8.11)
project (cuda_reaction_difusion LANGUAGES CXX CUDA)
add_executable (cuda_reaction_difusion 
  main.cu
)
set_target_properties(cuda_reaction_difusion
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cuda_reaction_difusion  
  -lpthread
)
