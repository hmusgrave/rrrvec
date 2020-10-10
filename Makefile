.DEFAULT_GOAL := test
.PHONY: build clean test

build: rrrvec.pyx setup.py
	python setup.py build_ext --inplace

clean:
	rm -rf build
	rm -f *.so
	rm -f *.c
	rm -rf zig-cache

test: build
	python main.py
