# Makefile for Linux NPU Test
TARGET = npu_test
SRC = main.c

CC = $(CROSS_COMPILE)gcc
CFLAGS = -Wall -O3 -g

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
