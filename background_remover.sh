#!/bin/sh
for f in *.png

do
  convert "$f" -fuzz 10% -transparent grey19 "$f"
  echo "Processing $f"
done