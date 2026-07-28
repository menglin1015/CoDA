#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

IMAGENET_FOLDER="/media/lm/_dde_data/data/imagewoof_in1k"
MODEL_FOLDER="$SCRIPT_DIR/../model/SDXL-Refiner"

run_experiment() {
    local run_step1=${1:-true}
    local flag_features=${2:-false}
    local flag_cluster=${3:-false}
    local flag_generate=${4:-false}
    local run_step2=${5:-true}

    local run_stages=""
    if [[ "$flag_features" == "true" ]]; then
        run_stages="$run_stages --calcu_features"
    fi
    if [[ "$flag_cluster" == "true" ]]; then
        run_stages="$run_stages --calcu_cluster"
    fi
    if [[ "$flag_generate" == "true" ]]; then
        run_stages="$run_stages --generate_images"
    fi

    if [[ "$run_step1" == "true" ]]; then

        python CoDA_main.py \
            --dataset_dir "$IMAGENET_FOLDER" --local_model_path "$MODEL_FOLDER" \
            --spec "$SPEC" \
            --IPC "$ipc" \
            --n_neighbors "$n_neighbors" --min_cluster_size "$size_min" \
            --cluster_detial --cluster_logger \
            --sample_step "$timestep" --denoising_factor "$DF" --guideTPercent "$GTP" --CoDA_guidance_scale "$gamma" \
            $run_stages

    fi

    if [[ "$run_step2" == "true" ]]; then

        local train_data_path="./results/${SPEC}/Step-${timestep}/IPC-${ipc}/DF-${DF}-GTP-${GTP}-gamma-${gamma}/n_${n_neighbors}_s_${size_min}"
        local val_data_path="$IMAGENET_FOLDER/validation"

        local use_real_images=${6:-true}
        local data_tag
        if [[ "$use_real_images" == "true" ]]; then
            train_data_path+="/real_images"
            data_tag="real"
        else
            train_data_path+="/generated_images"
            data_tag="gen"
        fi

        # data_tag 必须进 save-dir，否则 real / gen 两次评测会写进同一个目录互相覆盖
        local train_save_dir="./trained_results/${SPEC}/${data_tag}/ipc${ipc}/n_${n_neighbors}_s_${size_min}/step-$timestep-DF-$DF/GTP-$GTP-gamma-$gamma"

        echo "==> Testing with ResNet-AP 10..."
        python ./test/train.py --dataset_dir "$train_data_path" "$val_data_path" \
            -d imagenet --spec "$SPEC" --nclass 10  --size 256 --ipc "$ipc" \
            -n resnet_ap --depth 10  --save-dir "$train_save_dir-resnet_ap"  \
            --workers 8 \
            --n_neighbors "$n_neighbors" --min_cluster_size "$size_min" --tag test
    fi
}

export CUDA_VISIBLE_DEVICES=0

ipc=${IPC:-10}

n_neighbors=${N_NEIGHBORS:-85}
size_min=${SIZE_MIN:-55}

timestep=25
DF=1.0
GTP=0.9
gamma=0.05

SPEC_LIST="woof"
# Stage switches, overridable from the command line:
#   STEP1 / FEATURES / CLUSTER / GENERATE / STEP2 / REAL_IMAGES
STEP1=${STEP1:-true}
FEATURES=${FEATURES:-true}
CLUSTER=${CLUSTER:-true}
GENERATE=${GENERATE:-true}
STEP2=${STEP2:-true}
REAL_IMAGES=${REAL_IMAGES:-true}

for SPEC in $SPEC_LIST; do
    #                Step1    cal_features cal_cluster  generate     Step2     use_real_images
    run_experiment   "$STEP1" "$FEATURES"  "$CLUSTER"   "$GENERATE"  "$STEP2"  "$REAL_IMAGES"
done
