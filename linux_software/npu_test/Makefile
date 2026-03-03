# Makefile for Linux NPU Test
TARGET = npu_test
SRC = main.c

CC = $(CROSS_COMPILE)gcc
CFLAGS = -Wall -O3 -g

all: $(TARGET) ocm_test pp_test

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

ocm_test: ocm_test.c
	$(CC) $(CFLAGS) -o ocm_test ocm_test.c

pp_test: pp_test.c
	$(CC) $(CFLAGS) -o pp_test pp_test.c

clean:
	rm -f $(TARGET) ocm_test pp_test
