#!/bin/sh
for video in `ls videos`; do
  mkdir -p frames/$video
  ffmpeg -i videos/$video -vf fps=1 frames/$video/%04d.jpg
done
