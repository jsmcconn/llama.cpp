llama_build_and_test(test-arg-parser.cpp)
llama_build_and_test(test-model-resolution.cpp)
llama_build_and_test(test-kv-ssd-user-isolation.cpp)
# The residency tracker is intentionally an internal C++ component. Keep its
# white-box unit test out of Windows shared-library builds, where the
# implementation's internal llama symbols are not exported. Static builds
# link the implementation and llama directly so the test remains useful on
# Windows as well as Unix-like platforms.
if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    llama_build(test-moe-residency-routing.cpp ${PROJECT_SOURCE_DIR}/src/llama-moe-residency.cpp)
    target_include_directories(test-moe-residency-routing PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_test(test-moe-residency-routing)
endif()
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
