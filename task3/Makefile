NTESTS=100
L1=4096
L1A=4
LINE=64
DATA=../data/mnistcnn.bin

cachetests: classifier.c classifier.h libtests.a
	$(CC) -static -g -O3 -o $@ $^

.PHONY: run
run: cachetests
	valgrind -q --tool=cachegrind --D1=${L1},${L1A},${LINE} --cachegrind-out-file=cachetests.cgout ./cachetests -n $(NTESTS) $(DATA)
	cg_annotate --show-percs=no cachetests.cgout | tee cachetests.rep

.PHONY: clean
clean:
	$(RM) *.o *.cgout *.rep cachetests

