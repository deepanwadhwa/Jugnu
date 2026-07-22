#!/bin/sh
set -eu

GATEWAY=${SAMOSA_COMPILED_GATEWAY:-./samosa-gateway}
JOBSD=${SAMOSA_COMPILED_JOBSD:-./samosa-jobsd}
BACKEND=${SAMOSA_FAKE_BACKEND:-./test_fake_openai_backend}
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
FS_SIDECAR=${SAMOSA_FS:-"$ROOT/build/samosa-fs"}
EXTRACTOR=${SAMOSA_EXTRACT:-"$ROOT/build/samosa-extract"}
TMP=$(mktemp -d "${TMPDIR:-/tmp}/samosa-compiled-gateway.XXXXXX")
HOME_DIR="$TMP/home"
PORT=18977
BACKEND_PORT=18978
PID=""

cleanup() {
  [ -z "$PID" ] || kill "$PID" 2>/dev/null || true
  [ -z "$PID" ] || wait "$PID" 2>/dev/null || true
  /bin/rm -rf "$TMP"
}
trap cleanup EXIT HUP INT TERM

mkdir -p "$HOME_DIR/models/ornith-9b"
printf 'fixture\n' >"$HOME_DIR/models/ornith-9b/Ornith-1.0-9B-Q4_K_M.gguf"
printf 'ornith\n' >"$HOME_DIR/model-backend"
printf '<!doctype html><title>Compiled Samosa</title>\n' >"$TMP/app.html"
printf 'png\n' >"$TMP/logo.png"
/bin/mkdir "$TMP/files"
printf "Titli vaccination record, rabies booster 2026.\n" >"$TMP/files/cat-medical-note.txt"
printf "Miso vaccination record.\n" >"$TMP/files/miso-record.txt"
printf "Cafe total 4.50\n" >"$TMP/files/receipt-b.txt"
/bin/mkdir -p "$HOME_DIR/jobs/review-native/results"
printf 'Coffee Shop\nTotal 8.37\n' >"$TMP/files/receipt.txt"
printf '{"unit_id":"u1","status":"review_required","input_path":"%s","extracted":{"merchant":"Coffee","total":8.0}}\n' \
  "$TMP/files/receipt.txt" >"$HOME_DIR/jobs/review-native/results/output.jsonl"
printf '{"unit_id":"u2","status":"passed","extracted":{"merchant":"Done"}}\n' \
  >>"$HOME_DIR/jobs/review-native/results/output.jsonl"
/bin/mkdir "$TMP/slow"
printf '%s\n' '#!/bin/sh' \
  'last=""; for arg do last=$arg; done' \
  'case "$last" in' \
  '  */slow) printf "%s\\n" "$$" >"'$TMP'/slow-sidecar.pid"; exec /bin/sleep 30 ;;' \
  'esac' \
  'exec "'$FS_SIDECAR'" "$@"' >"$TMP/samosa-fs-wrapper"
/bin/chmod +x "$TMP/samosa-fs-wrapper"

# Deliberately expose no external executable through PATH. All utilities used
# below have absolute paths; the gateway/backend receive the same environment.
PATH="$TMP/no-python-bin"
/bin/mkdir "$PATH"
export PATH
if command -v python3 >/dev/null 2>&1; then
  echo "compiled gateway test PATH unexpectedly contains python3" >&2
  exit 1
fi

SAMOSA_HOME="$HOME_DIR" \
SAMOSA_PORT="$PORT" \
SAMOSA_BACKEND_PORT="$BACKEND_PORT" \
SAMOSA_APP_HTML="$TMP/app.html" \
SAMOSA_APP_LOGO="$TMP/logo.png" \
SAMOSA_BONSAI_SERVER="$BACKEND" \
SAMOSA_ORNITH_MODEL="$HOME_DIR/models/ornith-9b/Ornith-1.0-9B-Q4_K_M.gguf" \
SAMOSA_FS="$TMP/samosa-fs-wrapper" \
SAMOSA_EXTRACT="$EXTRACTOR" \
"$GATEWAY" >"$TMP/gateway.log" 2>&1 &
PID=$!

i=0
while [ "$i" -lt 100 ]; do
  health=$(/usr/bin/curl -fsS "http://127.0.0.1:$PORT/healthz" 2>/dev/null || true)
  printf '%s' "$health" | /usr/bin/grep -q '"ready":true' && break
  kill -0 "$PID" 2>/dev/null || { /bin/cat "$TMP/gateway.log" >&2; exit 1; }
  /bin/sleep 0.05
  i=$((i + 1))
done
printf '%s' "$health" | /usr/bin/grep -q '"compiled":true'
printf '%s' "$health" | /usr/bin/grep -q '"ready":true'

reply=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary '{"messages":[{"role":"user","content":"hello"}],"stream":false}')
printf '%s' "$reply" | /usr/bin/grep -q 'compiled reply'

report=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/run" \
  -H 'Content-Type: application/json' \
  --data-binary "{\"goal\":\"report what is here\",\"folder\":\"$TMP/files\"}")
printf '%s' "$report" | /usr/bin/grep -q '"type":"report"'
printf '%s' "$report" | /usr/bin/grep -q '"type":"done"'

find=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/run" \
  -H 'Content-Type: application/json' \
  --data-binary "{\"goal\":\"find my cat medical record\",\"folder\":\"$TMP/files\"}")
printf '%s' "$find" | /usr/bin/grep -q '"tool":"fs_read_text"'
printf '%s' "$find" | /usr/bin/grep -q "Found the matching record at cat-medical-note.txt"
if printf '%s' "$find" | /usr/bin/grep -q 'samosa_tool'; then
  echo "compiled find leaked tool protocol" >&2
  exit 1
fi

paused=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/run" \
  -H 'Content-Type: application/json' \
  --data-binary "{\"goal\":\"find a record\",\"folder\":\"$TMP/files\"}")
printf '%s' "$paused" | /usr/bin/grep -q '"type":"await_user"'
JOB_ID=$(printf '%s' "$paused" | /usr/bin/sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p' | /usr/bin/head -1)
[ -n "$JOB_ID" ]
resumed=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/answer" \
  -H 'Content-Type: application/json' \
  --data-binary "{\"job_id\":\"$JOB_ID\",\"answer\":\"Miso\"}")
printf '%s' "$resumed" | /usr/bin/grep -q "Found Miso's record at miso-record.txt"

review=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/review" \
  -H 'Content-Type: application/json' --data-binary '{"job_id":"review-native"}')
printf '%s' "$review" | /usr/bin/grep -q '"pending":1'
printf '%s' "$review" | /usr/bin/grep -q 'Coffee Shop'
corrected=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/review/correct" \
  -H 'Content-Type: application/json' \
  --data-binary '{"job_id":"review-native","item":{"unit_id":"u1"},"fields":{"merchant":"Coffee Shop","total":8.37}}')
printf '%s' "$corrected" | /usr/bin/grep -q '"pending":0'
/usr/bin/grep -q '"status":"passed"' "$HOME_DIR/jobs/review-native/results/output.jsonl"
/usr/bin/grep -q '"merchant":"Coffee Shop"' "$HOME_DIR/jobs/review-native/results/output.jsonl"
[ "$(/usr/bin/wc -l <"$HOME_DIR/jobs/review-native/results/output.jsonl" | /usr/bin/tr -d ' ')" = 2 ]

definition="{\"job\":{\"job_id\":\"native-definition\",\"input\":{\"folder\":\"$TMP/files\"},\"instruction\":\"Extract merchant and total.\",\"output_schema\":{\"type\":\"object\",\"properties\":{\"merchant\":{\"type\":\"string\"},\"total\":{\"type\":\"number\"}}},\"output\":{\"dir\":\"$TMP/definition-out\"}}}"
preview=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/definition/preview" \
  -H 'Content-Type: application/json' --data-binary "$definition")
printf '%s' "$preview" | /usr/bin/grep -q '"sample_count":1'
[ -f "$TMP/definition-out/preview/output.jsonl" ]
[ ! -f "$TMP/definition-out/output.jsonl" ]
expanded=$(printf '%s' "$definition" | /usr/bin/sed 's/}$/,"expanded":true}/')
preview3=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/definition/preview" \
  -H 'Content-Type: application/json' --data-binary "$expanded")
printf '%s' "$preview3" | /usr/bin/grep -q '"sample_count":3'
run=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/definition/run" \
  -H 'Content-Type: application/json' --data-binary "$definition")
printf '%s' "$run" | /usr/bin/grep -q '"type":"item_complete"'
printf '%s' "$run" | /usr/bin/grep -q '"type":"done"'
[ -f "$TMP/definition-out/output.jsonl" ]
/usr/bin/grep -q '"merchant":"Cafe"' "$TMP/definition-out/output.jsonl"

move_plan=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/run" \
  -H 'Content-Type: application/json' \
  --data-binary "{\"goal\":\"find cat medical record and move it to Archive\",\"folder\":\"$TMP/files\"}")
printf '%s' "$move_plan" | /usr/bin/grep -q '"type":"await_apply"'
MOVE_JOB=$(printf '%s' "$move_plan" | /usr/bin/sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p' | /usr/bin/tail -1)
applied=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/apply" \
  -H 'Content-Type: application/json' --data-binary "{\"job_id\":\"$MOVE_JOB\"}")
printf '%s' "$applied" | /usr/bin/grep -q '"applied":1'
[ -f "$TMP/files/Archive/cat-medical-note.txt" ]
[ ! -f "$TMP/files/cat-medical-note.txt" ]
undone=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/undo" \
  -H 'Content-Type: application/json' --data-binary "{\"job_id\":\"$MOVE_JOB\"}")
printf '%s' "$undone" | /usr/bin/grep -q '"undone":1'
[ -f "$TMP/files/cat-medical-note.txt" ]

# --- Native background scheduler: arm, idempotency, window/battery policy, jobsd binary ---
/bin/mkdir "$TMP/sched"
printf 'shift log entry\n' >"$TMP/sched/log-a.txt"
printf 'another note\n' >"$TMP/sched/log-b.txt"

# Arm an overnight (cross-midnight) report job.
SCHED_JOB="{\"job\":{\"job_id\":\"nightly-report\",\"input\":{\"folder\":\"$TMP/sched\"}},\"window_start\":\"22:00\",\"window_end\":\"06:00\",\"missed_policy\":\"skip\"}"
armed=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/schedule/arm" \
  -H 'Content-Type: application/json' --data-binary "$SCHED_JOB")
printf '%s' "$armed" | /usr/bin/grep -q '"ok":true'
printf '%s' "$armed" | /usr/bin/grep -q '"job_id":"nightly-report"'
[ -f "$HOME_DIR/jobs/nightly-report/schedule.json" ]
[ -f "$HOME_DIR/jobs/nightly-report/job.json" ]

# Re-arming the identical definition is idempotent (no rejection).
armed_again=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/schedule/arm" \
  -H 'Content-Type: application/json' --data-binary "$SCHED_JOB")
printf '%s' "$armed_again" | /usr/bin/grep -q '"ok":true'

# Arming a changed definition under the same job_id is rejected, not replaced.
CHANGED_JOB="{\"job\":{\"job_id\":\"nightly-report\",\"input\":{\"folder\":\"$TMP/sched\"},\"instruction\":\"different\"},\"window_start\":\"22:00\",\"window_end\":\"06:00\"}"
rejected=$(/usr/bin/curl -sS -X POST "http://127.0.0.1:$PORT/v1/jobs/schedule/arm" \
  -H 'Content-Type: application/json' --data-binary "$CHANGED_JOB")
printf '%s' "$rejected" | /usr/bin/grep -q '"code":"schedule_definition_changed"'

# Outside the window: defer.
outside=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobsd/once" \
  -H 'Content-Type: application/json' --data-binary '{"now_minutes":720,"on_battery":false}')
printf '%s' "$outside" | /usr/bin/grep -q '"job_id":"nightly-report","action":"defer","reason":"outside_window"'

# Inside the window but on battery (run_on_battery defaults false): defer.
battery=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobsd/once" \
  -H 'Content-Type: application/json' --data-binary '{"now_minutes":1380,"on_battery":true}')
printf '%s' "$battery" | /usr/bin/grep -q '"job_id":"nightly-report","action":"defer","reason":"on_battery"'

# Inside the window on AC: it runs to completion across midnight.
ran=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobsd/once" \
  -H 'Content-Type: application/json' --data-binary '{"now_minutes":1380,"on_battery":false}')
printf '%s' "$ran" | /usr/bin/grep -q '"job_id":"nightly-report","action":"run","reason":"inside_window"'
printf '%s' "$ran" | /usr/bin/grep -q '"status":"complete"'
/usr/bin/grep -q '"type":"scheduled_job_complete"' "$HOME_DIR/jobs/nightly-report/events.jsonl"

# One-shot polling is idempotent: a finished schedule does not run again.
again=$(/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobsd/once" \
  -H 'Content-Type: application/json' --data-binary '{"now_minutes":1380,"on_battery":false}')
printf '%s' "$again" | /usr/bin/grep -q '"job_id":"nightly-report","action":"defer"'

# The launchd plist points at the compiled samosa-jobsd one-shot.
plist=$(/usr/bin/curl -fsS "http://127.0.0.1:$PORT/v1/jobs/launchd-plist")
printf '%s' "$plist" | /usr/bin/grep -q 'samosa-jobsd'
printf '%s' "$plist" | /usr/bin/grep -q '<string>jobsd-once</string>'

# The standalone compiled daemon runs an armed job with python unavailable and no
# listener/backend. Arm a 24h window that ignores battery so it is time/power
# independent, then invoke the binary directly.
ALWAYS_JOB="{\"job\":{\"job_id\":\"always-report\",\"input\":{\"folder\":\"$TMP/sched\"},\"resources\":{\"run_on_battery\":true}},\"window_start\":\"00:00\",\"window_end\":\"00:00\",\"missed_policy\":\"skip\"}"
/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/jobs/schedule/arm" \
  -H 'Content-Type: application/json' --data-binary "$ALWAYS_JOB" | /usr/bin/grep -q '"ok":true'
jobsd_out=$(SAMOSA_HOME="$HOME_DIR" SAMOSA_FS="$TMP/samosa-fs-wrapper" "$JOBSD" jobsd-once)
printf '%s' "$jobsd_out" | /usr/bin/grep -q '"job_id":"always-report","action":"run"'
/usr/bin/grep -q '"type":"scheduled_job_complete"' "$HOME_DIR/jobs/always-report/events.jsonl"

/usr/bin/curl -sS -X POST "http://127.0.0.1:$PORT/v1/jobs/run" \
  -H 'Content-Type: application/json' \
  --data-binary "{\"goal\":\"report what is here\",\"folder\":\"$TMP/slow\"}" \
  >"$TMP/slow-result" 2>/dev/null &
SLOW_CURL=$!
i=0
while [ "$i" -lt 100 ] && [ ! -s "$TMP/slow-sidecar.pid" ]; do /bin/sleep 0.02; i=$((i + 1)); done
[ -s "$TMP/slow-sidecar.pid" ]
SIDE_PID=$(/bin/cat "$TMP/slow-sidecar.pid")
/usr/bin/curl -fsS -X POST "http://127.0.0.1:$PORT/v1/kill" >/dev/null
wait "$SLOW_CURL" 2>/dev/null || true
if /bin/kill -0 "$SIDE_PID" 2>/dev/null; then
  echo "kill route left a Jobs sidecar running" >&2
  exit 1
fi
wait "$PID"
PID=""
echo "compiled gateway without python: PASS"
