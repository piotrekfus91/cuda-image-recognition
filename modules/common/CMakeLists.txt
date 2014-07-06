add_library(common
src/common/MatWrapper.cpp 
src/common/ImageProcessingService.cpp
src/common/exception/InvalidMatTypeException.cpp
)

if(test)
	enable_testing()
	add_executable(RunTests test/common/MatWrapperTest.cpp)
	target_link_libraries(RunTests gtest gtest_main pthread)
	target_link_libraries(RunTests ${OpenCV_LIBS})
	target_link_libraries(RunTests common)
	add_test(Tests RunTests)
endif()