#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

: "${KOE_WHISPER_MODEL:?Set KOE_WHISPER_MODEL in .env}"

has_model=false
for arg in "$@"; do
  if [ "$arg" = "--model-trn" ]; then
    has_model=true
    break
  fi
done

model_arg=()
if [ "$has_model" = false ]; then
  model_arg=(--model-trn "$KOE_WHISPER_MODEL")
fi

exec cargo run -p koe-cli -- --asr whisper "${model_arg[@]}" "$@"
