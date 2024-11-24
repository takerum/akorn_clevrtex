#!/bin/bash
urls=(
    "https://thor.robots.ox.ac.uk/datasets/clevrtex/clevrtex_full_part1.tar.gz"
    "https://thor.robots.ox.ac.uk/datasets/clevrtex/clevrtex_full_part2.tar.gz"
    "https://thor.robots.ox.ac.uk/datasets/clevrtex/clevrtex_full_part3.tar.gz"
    "https://thor.robots.ox.ac.uk/datasets/clevrtex/clevrtex_full_part4.tar.gz"
    "https://thor.robots.ox.ac.uk/datasets/clevrtex/clevrtex_full_part5.tar.gz"
    "https://thor.robots.ox.ac.uk/datasets/clevrtex/clevrtex_outd.tar.gz"
    "https://thor.robots.ox.ac.uk/datasets/clevrtex/clevrtex_camo.tar.gz"
)

output_dir="data"

mkdir -p $output_dir

for url in "${urls[@]}"; do

    filename=$(basename "$url")

    echo "Downloading $filename..."
    wget -q --show-progress "$url" -P "$output_dir"

    echo "Extracting $filename..."
    tar -xzf "$output_dir/$filename" -C "$output_dir"

    rm "$output_dir/$filename"
done

echo "All files downloaded and extracted to $output_dir."