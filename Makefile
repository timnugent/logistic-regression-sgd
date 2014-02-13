all:
	g++ -Wall -O3 -std=c++11 lr_sgd.cpp -o lr_sgd 

clean:
	rm lr_sgd

test:
	./lr_sgd -i 1000 -o weights.out -p predict.out -t test.dat train.dat
