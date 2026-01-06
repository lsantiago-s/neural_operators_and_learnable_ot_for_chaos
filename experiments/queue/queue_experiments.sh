#!/usr/bin/env bash
# queue_experiments.sh
# Run a queue of experiments by cloning train.json with jq and launching main.py
set -euo pipefail

# ---------- Requirements ----------
command -v jq >/dev/null 2>&1 || { echo "jq is required (brew install jq)"; exit 1; }

# ---------- Paths ----------
BASE_DIR="/Users/ls/workspace/neural_operators_and_learnable_ot_for_chaos"
TRAIN_BASE="${BASE_DIR}/configs/train.json"
EVAL_CFG="${BASE_DIR}/configs/eval.json"

TS="$(date +"%Y%m%d_%H%M%S")"
SWEEP_DIR="${BASE_DIR}/configs/sweeps/${TS}"
LOG_DIR="${BASE_DIR}/logs/${TS}"
mkdir -p "${SWEEP_DIR}" "${LOG_DIR}"

# ---------- Optional W&B glue (edit or export before running) ----------
export WANDB_PROJECT="${WANDB_PROJECT:-chaos-emulator}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"               # empty ok
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-lorenz63}" # group to link runs
# you can append per-run tags below via WANDB_TAGS

# ---------- Helper: pick a free port ----------
pick_port() {
  while :; do
    port=$(awk -v min=49152 -v max=65535 'BEGIN{srand(); print int(min+rand()*(max-min+1))}')
    netstat -an 2>/dev/null | grep -q "[\.\:]$port " || break
  done
  echo "$port"
}

# ---------- GRID (edit this block) ----------
SEEDS=(0 1 2)
OT_PENALTIES=(1.0 0.0)        # goes to .train_config.ot_penalty
NOISE_LEVELS=(0.00 0.1 0.3)      # goes to .train_config.noise_level
SUMMARY_TYPES=("mlp" "identity" "projection")  # .train_config.summary_type
PROJECTION_STATES=(0 1 2)     # only used when summary_type=projection

# number of concurrent runs on local (macOS). 1 = strict queue.
MAX_PARALLEL="${MAX_PARALLEL:-1}"

# ---------- Helper: make a per-run train config with jq ----------
# Usage: make_cfg <base_json> <out_json> <seed> <ot_penalty> <noise> <summary_type> [proj_state]
make_cfg() {
  local in="$1"; shift
  local out="$1"; shift
  local seed="$1"; shift
  local ot="$1"; shift
  local noise="$1"; shift
  local summary="$1"; shift
  local jq_prog

  jq_prog="
    .train_config.seed = ${seed}
  | .train_config.ot_penalty = ${ot}
  | .train_config.noise_level = ${noise}
  | .train_config.summary_type = \"${summary}\"
  "

  if [[ "${summary}" == "projection" && $# -ge 1 ]]; then
    local state="$1"; shift
    jq_prog="${jq_prog} | .summary_config.state = ${state}"
  fi

  jq "${jq_prog}" "${in}" > "${out}"
}

# ---------- Runner ----------
run_single() {
  local cfg="$1"
  local run_name="$2"
  local log_file="$3"

  echo ">>> Launching ${run_name}"
  echo "Config: ${cfg}"
  echo "Log:    ${log_file}"

  # Per-run W&B tags (optional)
if [[ -n "${WANDB_TAGS:-}" ]]; then
  export WANDB_TAGS="${WANDB_TAGS},${run_name}"
else
  export WANDB_TAGS="${run_name}"
fi


  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Running on SLURM (job ${SLURM_JOB_ID})"

    export NCCL_DEBUG=INFO
    export NCCL_P2P_DISABLE=1
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_DISTRIBUTED_DEBUG=DETAIL

    MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    export MASTER_ADDR
    PORT=$(pick_port)
    echo "MASTER_ADDR=$MASTER_ADDR  PORT=$PORT"

    # sequential by default; if you want array jobs, submit this script via sbatch in a loop externally
    torchrun \
      --nproc_per_node=4 \
      --nnodes="${SLURM_NNODES}" \
      --rdzv_id="${SLURM_JOB_ID}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_ADDR}:${PORT}" \
      "${BASE_DIR}/main.py" \
      --train-config "${cfg}" \
      --eval-config "${EVAL_CFG}" \
      --run-eval \
      > "${log_file}" 2>&1

  elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running locally on macOS"
    python "${BASE_DIR}/main.py" \
      --train-config "${cfg}" \
      --eval-config "${EVAL_CFG}" \
      --run-eval \
      > "${log_file}" 2>&1
  else
    echo "Unsupported OS: ${OSTYPE}" && exit 1
  fi
}

# ---------- Orchestrate the sweep ----------
pids=()

for seed in "${SEEDS[@]}"; do
  for ot in "${OT_PENALTIES[@]}"; do
    for noise in "${NOISE_LEVELS[@]}"; do
      for summary in "${SUMMARY_TYPES[@]}"; do

        if [[ "${summary}" == "projection" ]]; then
          for state in "${PROJECTION_STATES[@]}"; do
            run_id="s${seed}_ot${ot}_n${noise}_${summary}-st${state}"
            cfg_path="${SWEEP_DIR}/train_${run_id}.json"
            log_path="${LOG_DIR}/${run_id}.log"
            make_cfg "${TRAIN_BASE}" "${cfg_path}" "${seed}" "${ot}" "${noise}" "${summary}" "${state}"

            # queue/run
            if [[ "${MAX_PARALLEL}" -gt 1 && "$OSTYPE" == "darwin"* ]]; then
              run_single "${cfg_path}" "${run_id}" "${log_path}" &
              pids+=($!)
              # throttle
              while ((${#pids[@]} >= MAX_PARALLEL)); do
                wait "${pids[0]}"; pids=("${pids[@]:1}")
              done
            else
              run_single "${cfg_path}" "${run_id}" "${log_path}"
            fi
          done
        else
          run_id="s${seed}_ot${ot}_n${noise}_${summary}"
          cfg_path="${SWEEP_DIR}/train_${run_id}.json"
          log_path="${LOG_DIR}/${run_id}.log"
          make_cfg "${TRAIN_BASE}" "${cfg_path}" "${seed}" "${ot}" "${noise}" "${summary}"

          if [[ "${MAX_PARALLEL}" -gt 1 && "$OSTYPE" == "darwin"* ]]; then
            run_single "${cfg_path}" "${run_id}" "${log_path}" &
            pids+=($!)
            while ((${#pids[@]} >= MAX_PARALLEL)); do
              wait "${pids[0]}"; pids=("${pids[@]:1}")
            done
          else
            run_single "${cfg_path}" "${run_id}" "${log_path}"
          fi
        fi

      done
    done
  done
done

# wait remaining background jobs (if any)
for pid in "${pids[@]:-}"; do wait "${pid}"; done

echo "All experiments finished. Logs in: ${LOG_DIR}"
echo "Configs saved in: ${SWEEP_DIR}"
