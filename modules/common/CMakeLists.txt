file(GLOB src_common_files "src/common/*.cpp")
file(GLOB src_common_exception_files "src/common/exception*.cpp")
file(GLOB test_common_files "test/common/*.cpp")

add_library(common
	${src_common_files}
	${src_common_exception_files}
)

if(test)
	add_executable(CommonTests
		${test_common_files}
	)
	
	target_link_libraries(CommonTests gtest gtest_main pthread)
	target_link_libraries(CommonTests ${OpenCV_LIBS})
	target_link_libraries(CommonTests common)
	
	add_test(Tests CommonTests)
endif()
