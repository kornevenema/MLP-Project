#!/bin/sh
for f in *.png
do
  wget -O background.png https://picsum.photos/128/128
  convert background.png "$f" -composite "$f"
done