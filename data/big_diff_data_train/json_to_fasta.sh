#!/bin/bash

# Parse command-line arguments
if [ $# -ne 2 ]; then
    echo "Usage: ./parse_json.sh <input_file> <output_file>"
    exit 1
fi

input_file=$1
output_file=$2

# Parse JSON data with jq and redirect output to awk
jq -r '.[] | ">" + .header + "\n" + .sequence' $input_file | awk '{printf "%s", $0} {if(NR%1==0) print ""}' > $output_file