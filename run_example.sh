#!/bin/bash

# PicoTron Tch Example Runner
# This script sets up the environment and runs the basic example

echo "Setting up environment for PicoTron Tch..."

# Set environment variables for tch-rs
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export LIBTORCH=/Users/somadhavala/work/mlsquare/rust-exps/eg01/picotron-tch/venv/lib/python3.13/site-packages/torch/lib
export DYLD_LIBRARY_PATH=/Users/somadhavala/work/mlsquare/rust-exps/eg01/picotron-tch/venv/lib/python3.13/site-packages/torch/lib:$DYLD_LIBRARY_PATH

echo "Environment variables set:"
echo "  LIBTORCH_USE_PYTORCH=$LIBTORCH_USE_PYTORCH"
echo "  LIBTORCH_BYPASS_VERSION_CHECK=$LIBTORCH_BYPASS_VERSION_CHECK"
echo "  LIBTORCH=$LIBTORCH"
echo "  DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"
echo ""

echo "Running PicoTron Tch basic example..."
echo "=================================="

# Run the example
cargo run --example basic_example

echo ""
echo "Example completed!"
