#!/usr/bin/env bash
# Download the real datasets used by phases 9-10.
# CharacterTrajectories (.mat) ships with the repo; this fetches it only if
# missing. UCI HAR (~60 MB) is always downloaded on demand (gitignored).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

if [ ! -f data/mixoutALL_shifted.mat ]; then
  echo "downloading CharacterTrajectories..."
  curl -sL -o data/mixoutALL_shifted.mat \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/character-trajectories/mixoutALL_shifted.mat"
fi

if [ ! -d "data/UCI HAR Dataset" ]; then
  echo "downloading UCI HAR (~60 MB)..."
  curl -sL -o data/har.zip \
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
  (cd data && unzip -qo har.zip && rm -rf __MACOSX har.zip)
fi

echo "done: $(ls data)"
