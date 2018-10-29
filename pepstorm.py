#!/bin/bash
cd "$(dirname "$0")"
ls
cd tests
ls
for folder in test_components test_generators
	do
	cd $folder
	autopep8 *.py --aggressive --in-place
	cd .. 
done
cd ../pyxem
for folder in components generators libraries signals utils  
	do 
	cd $folder
	autopep8 *.py --aggressive --in-place
	cd ..
done
 
	
