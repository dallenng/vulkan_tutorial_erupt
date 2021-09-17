#!/bin/bash

for shader in shaders/*.{vert,frag}; do
  glslc -O "$shader" -o "$shader".spv
done
