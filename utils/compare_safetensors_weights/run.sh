#!/bin/bash

home_dir="/home/robin/hfhub/models/moonshotai"

# 运行脚本
# 用法: ./compare_safetensors_parallel.sh <文件夹1路径> <文件夹2路径> <结果输出目录> [最大并发数]
bash compare_safetensors_parallel.sh \
    $home_dir/Kimi-K2-Base \
    $home_dir/Kimi-K2-Base-mcore-2-hf \
    ./comparison_logs 16
