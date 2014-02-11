all:
	g++ -Wall -O3 -std=c++11 lr_sgd.cpp -o lr_sgd 

clean:
	rm lr_sgd

test:
	./lr_sgd train.dat test.dat

