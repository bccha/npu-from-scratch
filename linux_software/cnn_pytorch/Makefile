CC = gcc
CFLAGS = -Wall -O3 -g

all: cnn_auto cnn_manual

cnn_auto: npu_auto_runtime.c npu_api.c
	$(CC) $(CFLAGS) -o cnn_auto npu_auto_runtime.c npu_api.c

cnn_manual: cnn_manual.c npu_api.c
	$(CC) $(CFLAGS) -o cnn_manual cnn_manual.c npu_api.c

clean:
	rm -f cnn_auto cnn_manual
