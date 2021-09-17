#!/bin/bash

texture_url="https://vulkan-tutorial.com/images/texture.jpg"
model_url="https://vulkan-tutorial.com/resources/viking_room.obj"
model_texture_url="https://vulkan-tutorial.com/resources/viking_room.png"

mkdir -p textures
mkdir -p models

if command -v wget; then
  wget -O textures/texture.jpg $texture_url
  wget -O textures/viking_room.png $model_texture_url
  wget -O models/viking_room.obj $model_url

  exit
fi

if command -v curl; then
  curl -o textures/texture.jpg $texture_url
  curl -o textures/viking_room.png $model_texture_url
  curl -o models/viking_room.obj $model_url

  exit
fi

echo "You need either wget or curl to download the assets, or do it manually"
echo $texture_url
echo $model_url
echo $model_texture_url
false
