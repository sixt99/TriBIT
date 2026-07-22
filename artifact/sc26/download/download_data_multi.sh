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

DATASETS_HYPERLINK=(
  #"https://data.dws.informatik.uni-mannheim.de/hyperlinkgraph/2014-03/webgraph/webgraph"
  #"https://data.dws.informatik.uni-mannheim.de/hyperlinkgraph/2012-08/webgraph/network"
)

OUTPUT_FOLDER="data/multi_gpu"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT_FOLDER="$2"
      shift 2
      ;;
    --output=*)
      OUTPUT_FOLDER="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

mkdir -p "$OUTPUT_FOLDER"

for name in "${DATASETS[@]}"; do
  echo "---------------------------------"
  echo "Downloading $name..."
  mkdir -p "${OUTPUT_FOLDER}/$(dirname "$name")"
  curl -f -C - -o "${OUTPUT_FOLDER}/${name}.properties" "${BASE_URL}/${name}.properties"
  curl -f -C - -o "${OUTPUT_FOLDER}/${name}.graph" "${BASE_URL}/${name}.graph"
done

for url in "${DATASETS_HYPERLINK[@]}"; do
  echo "---------------------------------"
  echo "Downloading $url..."
  name="$(basename "$url")"
  mkdir -p "${OUTPUT_FOLDER}/${name}"
  curl -f -C - -o "${OUTPUT_FOLDER}/${name}/${name}.properties" "${url}.properties"
  curl -f -C - -o "${OUTPUT_FOLDER}/${name}/${name}.graph" "${url}.graph"
done


