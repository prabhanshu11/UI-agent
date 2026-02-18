# cursor-daemon Makefile
# Build: make
# Clean: make clean

CC      = gcc
CFLAGS  = -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
          $(shell pkg-config --cflags libavformat libavcodec libavutil libswscale libavdevice)
LDFLAGS = $(shell pkg-config --libs libavformat libavcodec libavutil libswscale libavdevice) \
          -lturbojpeg -lpthread -lrt -lm

TARGET  = cursor-daemon
SRC     = cursor_daemon.c
HDR     = daemon_types.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)
