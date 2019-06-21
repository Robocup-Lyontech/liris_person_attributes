#!/bin/bash
#
# Transpose a 2 lines attribute-value file
# into a n lines attribute-value file
#
# Example:
#     attr1   attr2   attr3   ...
#     value1  value2  value3  ...
#   is transposed to
#     attr1  value1
#     attr2  value2
#     attr3  value3
#     ...    ...
#


if [ -z "$1" ]
then
  echo "Usage:  $0  2-lines-attribute-value-file"
  exit 1
fi


attributes="$(head -1 $1 | sed -e 's/  */ /g' -e 's/^ //')"
values="$(tail -1 $1 | sed -e 's/  */ /g' -e 's/^ //')"

#set -x

i=1
for attr in $attributes
do
  value="$(echo "$values" | cut -d\  -f$i)"
  echo "$attr $value"

  i="$(expr 1 + $i)"
done

#set +x
