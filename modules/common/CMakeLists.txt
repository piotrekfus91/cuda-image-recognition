add_library(common
src/common/MatWrapper.cpp 
src/common/ImageProcessingService.cpp
src/common/exception/InvalidMatTypeException.cpp
)

if(test)
	enable_testing()
	add_executable(CommonTests test/common/MatWrapperTest.cpp)
	target_link_libraries(CommonTests gtest gtest_main pthread)
	target_link_libraries(CommonTests ${OpenCV_LIBS})
	target_link_libraries(CommonTests common)
	add_test(Tests CommonTests)
endif()
