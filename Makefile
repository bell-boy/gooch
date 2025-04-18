CXX=g++
CXXFLAGS=-I. -std=c++20 -Wall -Wextra -pedantic -g
CC_FILES=${shell find . -name "*.cc"}
TEST_FILES=${shell find tests -name "*.cc"}
OBJS=${CC_FILES:.cc=.o}
TEST_OBJS=${TEST_FILES:.cc=.o}
HEADERS=${shell find . -name "*.h"}

all: ${OBJS} ${TEST_OBJS}

%.o: %.cc ${HEADERS} Makefile
	${CXX} ${CXXFLAGS} -c $< -o $@

%.test: tests/%.o ${OBJS} Makefile
	${CXX} ${CXXFLAGS} ${OBJS} -o $@

clean:
	rm -f ${OBJS} ${TEST_OBJS} *.test

