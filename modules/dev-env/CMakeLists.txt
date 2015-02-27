file(GLOB src_devenv_files "src/devenv/*.cpp")
file(GLOB src_devenv_cu_files "src/devenv/*.cu")
file(GLOB test_devenv_files "test/devenv/*.cpp")

add_library(dev-env
	${src_devenv_files}
	${src_devenv_cu_files}
)

find_package(CUDA)

if(test)
	enable_testing()
	
	cuda_add_executable(DevEnvTests
		${src_devenv_files}
		${src_devenv_cu_files}
		${test_devenv_files}
	)
	
	target_link_libraries(DevEnvTests gtest gtest_main pthread)
	target_link_libraries(DevEnvTests ${OpenCV_LIBS})
	target_link_libraries(DevEnvTests boost_thread boost_system)
	add_test(Tests DevEnvTests)
endif()