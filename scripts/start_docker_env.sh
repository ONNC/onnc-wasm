#!/bin/bash
mkdir -p $PWD/models
docker run -ti --rm -v $PWD:/home/onnc/workspace -v $PWD/models:/home/onnc/models onnc/onnc-wasm-backend-community