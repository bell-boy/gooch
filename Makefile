CXX=g++
CXXFLAGS=-I. -std=c++11 -Wall -Wextra -pedantic -g
CC_FILES=${shell find . -name "*.cc"}
TEST_FILES=${shell find tests -name "*.cc"}
OBJS=${CC_FILES:.cc=.o}
TEST_OBJS=${TEST_FILES:.cc=.o}

all: ${OBJS} ${TEST_OBJS}

%.o: %.cc
	${CXX} ${CXXFLAGS} -c $< -o $@

%.test: tests/%.o ${OBJS}
	${CXX} ${CXXFLAGS} $^ -o $@

clean:
	rm -f ${OBJS} ${TEST_OBJS} *.test

