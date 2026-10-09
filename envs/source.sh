#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${VMBMK_PYTHON:-python3}" "$root/source.py" "$@"
