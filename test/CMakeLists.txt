#Testing macro
macro (add_test_executable_with_lib _name libs)
	add_executable(test_${_name} "${_name}.cpp")
	target_link_libraries(test_${_name} ${EMP-ZK_LIBRARIES})
endmacro()

macro (add_test_case _name)
	add_test_executable_with_lib(${_name} "")
  	add_test(NAME ${_name} COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()

macro (add_test_case_with_run _name)
	add_test_executable_with_lib(${_name} "")
	add_test(NAME ${_name} COMMAND "./run" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")

endmacro()

# Test cases


add_test_case_with_run(basic_zk)
add_test_case_with_run(service_phase_bench)
add_test_case_with_run(audit_phase_fair_check_bench)
add_test_case_with_run(audit_phase_sensitive_attr_bench)
add_test_case_with_run(audit_phase_balanced_sample_bench)
add_test_case_with_run(audit_phase_correctness_check_bench)
add_test_case_with_run(sha256)
add_test_case_with_run(lowmc)
add_test_case_with_run(balanced_sample_unit)
add_test_case_with_run(unit_utils)
add_test_case_with_run(ro_ram_test)
add_test_case_with_run(nn_unit)
add_test_case_with_run(nn_baseline_bench)