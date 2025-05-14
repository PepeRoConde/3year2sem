#!/bin/bash

mkdir -p logs

# Fixed params
FIXED_ARGS="--anade_canny --tamano_batch 10 --procesos 7 --paciencia 60 --paciencia_paso 15 --razon 0.8"
SIZE="--novo_tamano 400 500"
DEVICE="--dispositivo cuda:0"
LR=1e-3

# Experiment grid
DEPTHS=(3 4 5 6)
DROPOUTS=(0.0 0.05 0.1)
BASE_CHANNELS=(8 16 32 64)
DATA_AUG=(true false)

run_id=0

for da in "${DATA_AUG[@]}"; do
  for prof in "${DEPTHS[@]}"; do
    for dropout in "${DROPOUTS[@]}"; do
      for base in "${BASE_CHANNELS[@]}"; do

        # Format DA flag
        AUG_FLAG=""
        if [ "$da" = "true" ]; then
          AUG_FLAG="--aumento_datos"
        fi

        # Unique log file
        log_name="exp_${run_id}_DA${da}_prof${prof}_drop${dropout}_base${base}.log"
        echo "▶️ Running experiment $run_id: DA=$da, depth=$prof, dropout=$dropout, base_channels=$base"

        python3 main.py \
          $FIXED_ARGS \
          $AUG_FLAG \
          $SIZE \
          $DEVICE \
          --profundidade $prof \
          --canles_base $base \
          --paso $LR \
          --factor_paso 0.7 \
          --probabilidade_dropout $dropout \
          --verboso \
          > logs/$log_name 2>&1

        echo "✅ Finished experiment $run_id — Log: logs/$log_name"
        ((run_id++))

      done
    done
  done
done

