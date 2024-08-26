#!/bin/bash

read -p "Hugging Face cache : " chosen_path

if [ -z "$chosen_path" ]; then
    echo "Choose a valid path"
    exit 1
fi

export HF_HOME="$chosen_path/.cache/huggingface"
export HF_DATASETS_CACHE="$chosen_path/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$chosen_path/.cache/huggingface/models"
