#!/bin/bash

for file in include/**/*.h
do
	echo "Processing $file..."
	# check whether "MIT License" is already present in file
	grep -q "MIT License" $file
	if [ $? -ne 0 ]
	then
		# if string is not present, append license information to file
    		cat LICENSE_HEADER "$file" > tempfile && mv tempfile $file
	fi
done

# uncomment this to remove license info
#for file in include/**/*.h
#do
#        echo "Processing $file..."
#        # check whether "MIT License" is already present in file
#        grep -q "MIT License" $file
#        if [ $? -eq 0 ]
#        then
#		sed -e '1,5d' < $file > tempfile && mv tempfile $file
#        fi
#done
