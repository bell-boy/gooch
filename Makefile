CXX := g++
CXXFLAGS := -std=c++20 -Wall -Wextra -Werror -I./src

BUILD_CC_FILES := ${wildcard src/*.cc}
BUILD_HEADERS := ${wildcard src/*.h}

TEST_CC_FILES := ${wildcard tests/*.cc}
TEST_OK_FILES := ${wildcard tests/*.ok}
TEST_HEADERS := ${wildcard tests/*.h}

TEST_RUNS := ${patsubst tests/%.cc,build/tests/%.run,${TEST_CC_FILES}}
TEST_OUTS := ${patsubst tests/%.cc,build/tests/%.out,${TEST_CC_FILES}}
TEST_RESULTS := ${patsubst tests/%.cc,build/tests/%.result,${TEST_CC_FILES}}
TEST_DIFFS := ${patsubst tests/%.cc,build/tests/%.diff,${TEST_CC_FILES}}

OBJS := ${patsubst src/%.cc,build/%.o,${BUILD_CC_FILES}}
TEST_OBJS := ${patsubst tests/%.cc,build/tests/%.o,${TEST_CC_FILES}}

${OBJS}: build/%.o: src/%.cc ${BUILD_HEADERS} ${TEST_HEADERS} Makefile
	mkdir -p $(dir $@)
	${CXX} ${CXXFLAGS} -c $< -o $@

${TEST_OBJS}: build/tests/%.o: tests/%.cc ${BUILD_HEADERS} ${TEST_HEADERS} Makefile
	mkdir -p $(dir $@)
	${CXX} ${CXXFLAGS} -c $< -o $@

${TEST_RUNS}: build/tests/%.run: build/tests/%.o ${OBJS} Makefile
	${CXX} ${CXXFLAGS} ${OBJS} $< -o $@

${TEST_OUTS}: build/tests/%.out: build/tests/%.run ${TEST_HEADERS} Makefile
	@{ /usr/bin/time -p $< > $@; } 2> build/tests/$*.time

${TEST_RESULTS}: build/tests/%.result: build/tests/%.out ${TEST_HEADERS} ${TEST_OK_FILES} Makefile
	@diff -u ${patsubst build/tests/%.out,tests/%.ok, $<} $< > build/tests/$*.diff || (cp build/tests/$*.diff $@ && echo "FAIL" > $@)
	@if [ ! -s build/tests/$*.diff ]; then echo "PASS" > $@; fi

.PHONY: test
test: ${TEST_RUNS} ${TEST_OUTS} ${TEST_RESULTS}
	@for result in ${TEST_RESULTS}; do \
		test_name=$$(basename $${result} .result); \
		if grep -q "PASS" "$${result}"; then \
			time_val=$$(cat build/tests/$${test_name}.time | grep "real" | awk '{print $$2}'); \
			echo "$${test_name}: pass [$${time_val}s]"; \
		else \
			time_val=$$(cat build/tests/$${test_name}.time | grep "real" | awk '{print $$2}'); \
			echo "$${test_name}: fail [$${time_val}s]"; \
		fi; \
	done

.PHONY: clean
clean:
	rm -rf build
