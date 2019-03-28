
all: SRAD_V1 SRAD_V2

SRAD_V1:
	cd srad_v1; 	make; cd ..; cp srad_v1/srad sradv1

SRAD_V2:
	cd srad_v2; 	make; cd ..; cp srad_v2/srad sradv2


clean: SRAD_V1_clean SRAD_V2_clean

SRAD_V1_clean:
	cd srad_v1; 	make clean; cd ..; rm -f sradv1
	
SRAD_V2_clean:
	cd srad_v2; 	make clean; cd ..;rm -f sradv2



