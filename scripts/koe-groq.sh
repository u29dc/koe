#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

: "${GROQ_API_KEY:?Set GROQ_API_KEY in .env}"

exec cargo run -p koe-cli -- --asr groq "$@"
