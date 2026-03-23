CC = gcc
CFLAGS = -Wall -O3 -g

all: mnist_test mnist_manual mnist_auto

mnist_test: main.c
	$(CC) $(CFLAGS) -o mnist_test main.c

mnist_manual: mnist_manual.c npu_api.c
	$(CC) $(CFLAGS) -o mnist_manual mnist_manual.c npu_api.c

mnist_auto: npu_auto_runtime.c npu_api.c
	$(CC) $(CFLAGS) -o mnist_auto npu_auto_runtime.c npu_api.c

clean:
	rm -f mnist_test mnist_manual mnist_auto
