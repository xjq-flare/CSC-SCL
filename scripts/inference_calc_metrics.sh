#!/bin/bash
# $1: experiment name
# $2: dataset root (e.g. UIEB_HCLR)
# $3: args pass to test
# $4 args pass to calc_metrics (e.g. --noref)
set -ex

 gpu_ids=0
 export CUDA_VISIBLE_DEVICES=$gpu_ids

if [ $# -eq 0 ]; then
    echo "please input the name and the dataroot"
    exit 1
else
    name=$1
fi

if [ -n "$2" ]; then
    dataroot=$2
else
    echo "please input dataset root"
    exit 1
fi

result_dir="./results"
record_file="$result_dir/$name/test_record_$name.txt"
mkdir -p "$result_dir/$name"
input="256"
epochs=("800")

echo "testing $dataroot" | tee -a "$record_file"
for epoch in "${epochs[@]}"
do
  echo "Execution Time: $(date +"%Y-%m-%d %H:%M:%S")" >> "$record_file"
  echo "$name epoch: $epoch | input model: $input | calc metrics 256:" | tee -a "$record_file"
  python test.py --dataroot ./datasets/"$dataroot"/testA --name "$name" --model test \
  --load_size $input --preprocess resize --dataset_mode single --model_suffix _A --no_dropout \
  --epoch "$epoch" --results_dir "$result_dir" --gpu_ids $gpu_ids --netG 'resnet_9blocks_cc_up_sc' $3 | tee -a "$record_file"
  python calc_metrics.py --gen $result_dir/"$name"/test_"$epoch"/images \
    --gt ./datasets/"$dataroot"/testB_gt \
    --single $4 | tee -a "$record_file"
done
