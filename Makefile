# Target Platform: Linux

TARGET = isolhom
BUILD_DIR = build

SOURCES = src/main.cpp
SOURCES += src/configraw/configraw.cpp
SOURCES += src/calc/lhom.cu
SOURCES += src/progbar/progbar.hpp

OBJS = $(addsuffix .o, $(basename $(SOURCES)))

INCLUDE = -I./src -I./src/configraw
INCLUDE += -I./include -I./lib/gnuplot-iostream -I./lib/nlohmann -I./lib/CImg
INCLUDE += -I/usr/local/cuda/include
LDINCLUDE = -L/opt/cuda/targets/x86_64-linux/lib

CC = cc
CXX = g++
CUDAC = nvcc
LINKER = ld

CXXFLAGS = $(INCLUDE) -std=c++14 -fopenmp
CXXFLAGS += -Wall -Wextra
DEBUG_CXXFLAGS = -DDEBUG -g
RELEASE_CXXFLAGS = -DRELEASE -O3

CFLAGS = $(INCLUDE) -std=c11 -fopenmp
CFLAGS += -Wall -Wextra
DEBUG_CFLAGS = -DDEBUG -g
RELEASE_CFLAGS = -DRELEASE -O3

CUDACFLAGS = $(INCLUDE) -std=c++14 --gpu-architecture=compute_30
DEBUG_CUDACFLAGS = -DDEBUG -g
RELEASE_CUDACFLAGS = -DRELEASE -O3

LDFLAGS = $(LDINCLUDE) -fopenmp
LDFLAGS += -lX11
# LDFLAGS += -lm -lpthread -lX11
LDFLAGS += -lcuda -lcudart
LDFLAGS += -lboost_system -lboost_filesystem -lboost_regex
LDFLAGS += -lboost_iostreams -lboost_program_options

.PHONY: clean

default: debug

debug: CADDITIONALFLAGS = $(DEBUG_CFLAGS)
debug: CXXADDITIONALFLAGS = $(DEBUG_CXXFLAGS)
debug: CUDACADDITIONALFLAGS = $(DEBUG_CUDACFLAGS)
debug: TARGET_DIR = $(BUILD_DIR)/debug
debug: $(BUILD_DIR) $(BUILD_DIR)/debug start $(TARGET)
	@echo Build of standalone executable complete!

release: CADDITIONALFLAGS = $(RELEASE_CFLAGS)
release: CXXADDITIONALFLAGS = $(RELEASE_CXXFLAGS)
release: CUDACADDITIONALFLAGS = $(RELEASE_CUDACFLAGS)
release: TARGET_DIR = $(BUILD_DIR)/release
release: $(BUILD_DIR) $(BUILD_DIR)/release start $(TARGET)
	@echo Build of standalone executable complete!

start:
	@echo -------------------------------------------------------------------------------
	@echo Compiling...
	@echo
	@echo CXXFLAGS: $(CXXFLAGS) $(CXXADDITIONALFLAGS)
	@echo CFLAGS: $(CFLAGS) $(CADDITIONALFLAGS)
	@echo CUDACFLAGS: $(CUDACFLAGS) $(CUDACADDITIONALFLAGS)
	@echo TARGET_DIR $(TARGET_DIR)
	@echo

%.o: %.cpp
	@echo $<
	@$(CXX) $(CXXFLAGS) $(CXXADDITIONALFLAGS) -c -o $(TARGET_DIR)/$(@F) $<

%.o: %.c
	@echo $<
	@$(CC) $(CFLAGS) $(CADDITIONALFLAGS) -c -o $(TARGET_DIR)/$(@F) $<

%.o: %.cu
	@echo $<
	@$(CUDAC) $(CUDACFLAGS) $(CUDACADDITIONALFLAGS) -c -o $(TARGET_DIR)/$(@F) $<

$(BUILD_DIR):
	@echo Creating build directory...
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%:
	@echo Creating target directory...
	@mkdir -p $@

$(TARGET): $(OBJS)
	@echo -------------------------------------------------------------------------------
	@echo Linking executable...
	@echo
	@echo LDFLAGS: $(LDFLAGS) $(LDADDITIONALFLAGS)
	@echo TARGET_DIR: $(TARGET_DIR)
	@echo TARGET: $(TARGET)
	@echo
	@$(CXX) $(addprefix $(TARGET_DIR)/, $(notdir $^)) $(LDFLAGS) $(LDADDITIONALFLAGS) -o $(TARGET_DIR)/$(TARGET)

clean:
	@echo Cleaning up...
	@rm -rf ./$(BUILD_DIR)/debug
	@rm -rf ./$(BUILD_DIR)/release
	@echo Done!

