#!/usr/bin/env bash
DATASETS=(
  "twitter-2010/twitter-2010"
  "gsh-2015-host/gsh-2015-host"
  "uk-2006-09/uk-2006-09"
  "uk-2007-01/uk-2007-01"
  "uk-2007-02/uk-2007-02"
  "uk-union-2006-06-2007-05/uk-union-2006-06-2007-05-underlying"
  "gsh-2015/gsh-2015"
  "uk-2014/uk-2014"
  "eu-2015/eu-2015"
)
BASE_URL="http://data.law.di.unimi.it/webdata"
OUTPUT_FOLDER="../data/multi_gpu"

mkdir -p "$OUTPUT_FOLDER"

for name in "${DATASETS[@]}"; do
  echo "---------------------------------"
  echo "Downloading $name..."
  mkdir -p "${OUTPUT_FOLDER}/$(dirname "$name")"
  curl -f -C - -o "${OUTPUT_FOLDER}/${name}.properties" "${BASE_URL}/${name}.properties"
  curl -f -C - -o "${OUTPUT_FOLDER}/${name}.graph" "${BASE_URL}/${name}.graph"
done
