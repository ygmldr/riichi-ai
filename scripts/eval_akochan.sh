#!/usr/bin/env bash
# 训练里程碑: akochan 基准评估(绝对参照 / 趋势尺)
#
# 用法: bash scripts/eval_akochan.sh [games]
#   games 默认 40(趋势观察建议 40~100;须为 4 的倍数 —— 1 set = 同一 seed 的
#   a/b/c/d 四种座位 = 4 局)。
#
# 设计(2026-09-17 起,对齐 todo 3 的 12000 局协议;不再硬编码 2000/-1):
#   * 只读仓库配置 configs/config.toml:从 [1v3] 读 games_per_iter / seed_key /
#     seed_start,从 [1v3.challenger] 读 state_file —— 绝不改动仓库配置本身;
#   * 按分块(chunk,默认 4 局 = 1 set)调用 arena,而不是一次性把全部 games 塞进
#     一个 batch。原因:akochan 二进制存在**确定性 bad_alloc 毒局**(见
#     runs/akochan_eval/TREND.md),而 arena 只在整批对局结束后一次性落盘 ——
#     单批 40 局一旦撞上毒局则数小时计算全部丢失、0 个牌谱。分块后:每块独立
#     落盘,毒块只损失 4 局;脚本记录失败窗口并用**后续** seed 窗口重试(不重试
#     同一窗口),直到凑满 games 局。失败窗口全部写日志,绝不静默吞掉;
#   * 默认并行 (AKO_PARALLEL,默认 4) 个 worker,各自独立 seed 区间/子目录/日志,
#     结束后合并子目录并校验总数 —— 因为毒局率实测可到 ~20-70%(局部 seed 区间),
#     串行重试太慢。每 worker 的尝试次数有界 (4*chunks_k+8)。
#       分块大小: AKO_CHUNK_GAMES(默认 4,必须是 4 的倍数)
#   * 生成临时配置 /tmp/opencode/akochan_eval/config.at-<tag>_seed<S>_<n>g.toml
#     并以 MORTAL_CFG 注入,其中:
#       - [1v3.akochan].enabled   = true
#       - [1v3.challenger].state_file = 快照(不可变)
#       - [1v3].games_per_iter    = 本块局数
#       - [1v3].iters             = 1
#       - [1v3].seed_key/seed_start = 读到的值(可用环境变量覆盖)
#       - [1v3].log_dir           = runs/akochan_eval/games(或 worker 子目录)
#         (独立目录 —— 绝不写 runs/1v3 冻结语料,也绝不写 runs/variance)
#     生成后逐项读回断言(断言"实际注入的值",不是旧脚本那种写死的 2000/-1);
#   * 快照挑战者 -> runs/akochan_eval/mortal.pth.at-<tag>(已存在则复用):
#     默认挑战者 = [1v3.challenger].state_file(计划决策 #4:判决性评估用钉住的
#     runs/exp/PINNED/mortal.pth.at-175200;live checkpoints/mortal.pth 只配冒烟);
#   * 直接调用 vendor/mortal/mortal/one_vs_three.py(不走 run_eval.sh:
#     run_eval.sh:12 硬编码 LOG_DIR=runs/1v3,会往冻结语料目录追加 eval_*.log);
#   * 最后 python3 scripts/parse_eval.py <合并日志> --akochan,输出 "我方每席位均分"。
#
# 环境变量(可选,趋势尺用):
#   AKO_SEED_START  覆盖 seed_start(默认取 [1v3].seed_start);趋势尺窗口起点,
#                   按块连续消费 seed,失败块自动顺延
#   AKO_SEED_KEY    覆盖 seed_key(int(x,0);默认取 [1v3].seed_key)
#   AKO_CHALLENGER  覆盖挑战者权重文件(默认取 [1v3.challenger].state_file)
#   AKO_CHUNK_GAMES 分块大小(默认 4)
#   AKO_PARALLEL    并行 worker 数(默认 4;1 = 纯串行)
#   AKO_SEED_STRIDE 并行时 worker k 的 seed 起点 = SEED_START + k*STRIDE(默认 1000)
#   AKO_GAMES_DIR   牌谱输出目录(默认 runs/akochan_eval/games)
#   AKO_DRY_RUN=1   只生成并校验第一块的临时配置,不跑对局(用于自检)
#
# 速度: RUNBOOK:147-149 记 "1 set ≈13.5 分钟"(2026-09-16 观测,训练并发时);
#   40 局 = 10 块。实际吞吐以运行日志 + runs/akochan_eval/TREND.md 为准。
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_CFG="$REPO_ROOT/configs/config.toml"
SNAP_DIR="$REPO_ROOT/runs/akochan_eval"
TMP_DIR="/tmp/opencode/akochan_eval"
GAMES_ARG="${1:-}"
CHUNK="${AKO_CHUNK_GAMES:-4}"
PAR="${AKO_PARALLEL:-4}"
SEED_STRIDE="${AKO_SEED_STRIDE:-1000}"

if [[ -n "$GAMES_ARG" ]] && ! [[ "$GAMES_ARG" =~ ^[0-9]+$ ]]; then
    echo "ERROR: games 必须是整数: $GAMES_ARG" >&2
    exit 1
fi
if ! [[ "$CHUNK" =~ ^[0-9]+$ ]] || (( CHUNK <= 0 || CHUNK % 4 != 0 )); then
    echo "ERROR: AKO_CHUNK_GAMES 必须是正的 4 的倍数: $CHUNK" >&2
    exit 1
fi
if ! [[ "$PAR" =~ ^[0-9]+$ ]] || (( PAR <= 0 )); then
    echo "ERROR: AKO_PARALLEL 必须是正整数: $PAR" >&2
    exit 1
fi

# 环境(需要 python3 >= 3.11 的 tomllib;activate_env.sh 会把 python3 指到 .venv)
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/activate_env.sh"

declare -a GENERATED=()  # [dst, snap, games_dir, games, key_hex, seed_start, challenger, tag]
make_chunk_cfg() {        # make_chunk_cfg <games> <seed_start> [games_dir_override]
    mapfile -t -d '' GENERATED < <(python3 - "$SRC_CFG" "$TMP_DIR" "$SNAP_DIR" "$1" \
        "${AKO_SEED_KEY:-}" "$2" "${AKO_CHALLENGER:-}" "$REPO_ROOT" "${3:-${AKO_GAMES_DIR:-}}" <<'PYEOF'
import os
import re
import shutil
import sys
import tomllib

(src, tmp_dir, snap_dir, games_arg, seed_key_arg, seed_start_arg, challenger_arg, repo_root,
 games_dir_arg) = sys.argv[1:10]


def rewrite_section(text, section, updates):
    """在 `[section]` 段内做 `key = value` 行替换(保留其它行/注释),返回新文本。"""
    lines = text.splitlines(keepends=True)
    out, seen, in_sec = [], set(), False
    for line in lines:
        s = line.strip()
        if s.startswith("[") and "]" in s:
            in_sec = s == f"[{section}]"
            out.append(line)
            continue
        if in_sec:
            for k, v in updates.items():
                if re.match(rf"^{re.escape(k)}\s*=", s):
                    out.append(f"{k} = {v}\n")
                    seen.add(k)
                    line = None
                    break
            if line is None:
                continue
        out.append(line)
    missing = set(updates) - seen
    assert not missing, f"[{section}] 缺少待注入的键: {sorted(missing)}"
    return "".join(out)


with open(src, "rb") as f:
    src_text = f.read().decode("utf-8")
src_cfg = tomllib.loads(src_text)["1v3"]

games = int(games_arg)
if games <= 0 or games % 4 != 0:
    sys.exit(f"ERROR: games 必须是正的 4 的倍数: {games}")
seed_key = int(seed_key_arg, 0) if seed_key_arg else int(src_cfg["seed_key"])
seed_start = int(seed_start_arg, 0) if seed_start_arg else int(src_cfg.get("seed_start", 10000))
challenger = challenger_arg or src_cfg["challenger"]["state_file"]

if not os.path.isfile(challenger):
    sys.exit(f"ERROR: 挑战者权重不存在: {challenger}")

base = os.path.basename(challenger)
m = re.search(r"\.at-(\d+)$", base)
if m:
    tag = m.group(1)
else:
    tag = ""
    dqn_log = os.path.join(repo_root, "runs", "dqn_train.log")
    if os.path.isfile(dqn_log):
        with open(dqn_log, "r", errors="replace") as f:
            steps = re.findall(r"total steps: ([0-9,]+)", f.read())
        if steps:
            tag = steps[-1].replace(",", "")
    if not tag:
        tag = "unknown"

os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(snap_dir, exist_ok=True)
games_dir = games_dir_arg or os.path.join(snap_dir, "games")
os.makedirs(games_dir, exist_ok=True)
snap = os.path.join(snap_dir, f"mortal.pth.at-{tag}")

if not os.path.exists(snap):
    shutil.copy2(challenger, snap)
    print(f">>> 快照: {challenger} -> {snap}", file=sys.stderr)
elif os.path.getsize(snap) != os.path.getsize(challenger):
    print(f">>> 警告: 快照与来源大小不同(来源={challenger}),评估仍用快照", file=sys.stderr)

text = src_text
text = rewrite_section(text, "1v3.akochan", {"enabled": "true"})
text = rewrite_section(text, "1v3.challenger", {"state_file": f'"{snap}"'})
text = rewrite_section(
    text,
    "1v3",
    {
        "games_per_iter": str(games),
        "iters": "1",
        "seed_key": f"0x{seed_key:x}",
        "seed_start": str(seed_start),
        "log_dir": f'"{games_dir}"',
    },
)
dst = os.path.join(tmp_dir, f"config.at-{tag}_seed{seed_start}_{games}g.toml")
with open(dst, "w", encoding="utf-8") as f:
    f.write(text)

# 读回断言:断言"实际注入的值"(旧脚本断言的 games_per_iter=2000 / seed_key=-1 会被
# todo 3 的新值打破,故改为断言注入结果本身)。
with open(dst, "rb") as f:
    out = tomllib.loads(f.read().decode("utf-8"))["1v3"]
assert out["games_per_iter"] == games, out["games_per_iter"]
assert out["iters"] == 1, out["iters"]
assert out["seed_key"] == seed_key, out["seed_key"]
assert out["seed_start"] == seed_start, out["seed_start"]
assert out["log_dir"] == games_dir, out["log_dir"]
assert out["akochan"]["enabled"] is True
assert out["challenger"]["state_file"] == snap, out["challenger"]["state_file"]
assert out["akochan"]["dir"] == src_cfg["akochan"]["dir"]
assert out["akochan"]["tactics"] == src_cfg["akochan"]["tactics"]

print(f">>> 本次分块: games={games} seed_key=0x{seed_key:x} seed_start={seed_start} "
      f"窗口=[{seed_start},{seed_start + games // 4})", file=sys.stderr)
print(f">>> 对手: akochan x1(挑战者席位,{out['akochan']['dir']})| 我方 x3: {snap}", file=sys.stderr)
print(f">>> 牌谱目录: {games_dir}(独立目录,不碰 runs/1v3 / runs/variance)", file=sys.stderr)
print(f">>> 临时配置: {dst}(读回断言通过)", file=sys.stderr)

sys.stdout.write("".join(v + "\0" for v in
                         [dst, snap, games_dir, str(games), f"0x{seed_key:x}", str(seed_start), challenger, tag]))
PYEOF
    )
    if [[ ${#GENERATED[@]} -ne 8 ]]; then
        echo "ERROR: 配置生成失败(见上方 python 报错)" >&2
        exit 1
    fi
}

# 局数与 seed 起点
if [[ -z "$GAMES_ARG" ]]; then
    GAMES=$(python3 -c "import tomllib,sys; print(tomllib.load(open(sys.argv[1],'rb'))['1v3']['games_per_iter'])" "$SRC_CFG")
else
    GAMES="$GAMES_ARG"
fi
if ! [[ "$GAMES" =~ ^[0-9]+$ ]] || (( GAMES <= 0 || GAMES % 4 != 0 )); then
    echo "ERROR: games 必须是正的 4 的倍数: $GAMES" >&2
    exit 1
fi
SEED_START="${AKO_SEED_START:-$(python3 -c "import tomllib,sys; print(tomllib.load(open(sys.argv[1],'rb'))['1v3'].get('seed_start',10000))" "$SRC_CFG")}"
TOTAL_CHUNKS=$(( GAMES / 4 ))
(( PAR > TOTAL_CHUNKS )) && PAR="$TOTAL_CHUNKS"

# 第一块:生成快照与主目录,并取得 tag/games_dir/key(摘要由 python 打到 stderr)
FIRST_N=$CHUNK
(( FIRST_N > GAMES )) && FIRST_N=$GAMES
make_chunk_cfg "$FIRST_N" "$SEED_START"
GAMES_DIR="${AKO_GAMES_DIR:-${GENERATED[2]}}"
SNAP="${GENERATED[1]}"
SEED_KEY="${GENERATED[4]}"
TAG="${GENERATED[7]}"
KEY_DEC=$(python3 -c "print(int('$SEED_KEY', 0))")

TS="$(date +%Y%m%d_%H%M)"
RUN_LOG="$SNAP_DIR/akochan_eval_at-${TAG}_${GAMES}g_seed${SEED_START}_${TS}.log"
REPORT="$SNAP_DIR/parse_at-${TAG}_${GAMES}g_seed${SEED_START}_${TS}.txt"
WINDOWS="$SNAP_DIR/windows_at-${TAG}_${GAMES}g_seed${SEED_START}_${TS}.txt"

echo ">>> 计划: ${GAMES} 局 = ${TOTAL_CHUNKS} 块 x ≤${CHUNK} 局 | 并行 worker=${PAR} (chunk=${CHUNK})"
echo ">>> seed 起点=${SEED_START} key=0x$(printf '%x' "$KEY_DEC") 快照=$(basename "$SNAP")"
echo ">>> 合并日志: $RUN_LOG"

if [[ "${AKO_DRY_RUN:-0}" == "1" ]]; then
    echo ">>> AKO_DRY_RUN=1: 第一块临时配置已生成并通过读回断言,未运行对局"
    exit 0
fi

mkdir -p "$GAMES_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
echo ">>> OMP_NUM_THREADS=$OMP_NUM_THREADS (实测 akochan 单线程反而更快: 多线程同步开销大)"
echo ">>> 注意: WSL 下 SIGSTOP 无法可靠暂停训练,评估与训练并发运行(评估期间训练会变慢)"
echo ">>> akochan 基准评估开始: $(date '+%F %T')"

run_chunk() {  # run_chunk <games> <seed_start> <games_dir> <logfile> -> 0/非0
    local n="$1" seed="$2" dir="$3" log="$4"
    make_chunk_cfg "$n" "$seed" "$dir"
    local cfg="${GENERATED[0]}"
    local rc=0
    (
        cd "$REPO_ROOT/vendor/mortal/mortal"
        MORTAL_CFG="$cfg" python3 one_vs_three.py
    ) >> "$log" 2>&1 || rc=$?
    return "$rc"
}

run_worker() {  # run_worker <k> <target_games> <seed_start> <games_dir> <log>
    local k="$1" target="$2" cursor="$3" dir="$4" log="$5"
    local done_g=0 attempts=0 max_a=$(( 4 * (target / CHUNK) + 8 ))
    local -a used=() failed=()
    mkdir -p "$dir"
    : > "$log"
    while (( done_g < target && attempts < max_a )); do
        local n="$CHUNK"
        (( n > target - done_g )) && n=$(( target - done_g ))
        attempts=$(( attempts + 1 ))
        local cs="$cursor" ce=$(( cursor + n / 4 ))
        cursor="$ce"
        local before after got
        before=$(ls "$dir" | wc -l)
        echo ">>> [w${k} 尝试${attempts}] 块 ${n} 局 seed=[${cs},${ce}) $(date '+%F %T')" >&2
        if ! run_chunk "$n" "$cs" "$dir" "$log"; then
            echo "!!! [w${k}] 块失败 seed=[${cs},${ce}) —— 换后续窗口重试" >&2
            failed+=("[${cs},${ce})")
            continue
        fi
        after=$(ls "$dir" | wc -l)
        got=$(( after - before ))
        if (( got != n )); then
            echo "ERROR: [w${k}] 块退出码 0 但只落盘 ${got}/${n} 局(seed=[${cs},${ce}))" >&2
            return 1
        fi
        done_g=$(( done_g + got ))
        used+=("[${cs},${ce})")
        echo ">>> [w${k}] 块 OK +${got} 局,累计 ${done_g}/${target} $(date '+%F %T')" >&2
    done
    {
        echo "# worker ${k}: target=${target} seed_start=$3 done=${done_g}/${target} attempts=${attempts}"
        echo "成功窗口:"
        printf '  %s\n' "${used[@]:-}"
        echo "失败窗口:"
        printf '  %s\n' "${failed[@]:-}"
    } > "${log}.windows"
    (( done_g >= target ))
}

if (( PAR == 1 )); then
    declare -a USED=() FAILED=()
    : > "$RUN_LOG"
    # 串行:把 target 视作整体,用 run_worker 的逻辑循环
    N_REMAIN=$GAMES
    CURSOR="$SEED_START"
    ATTEMPTS=0
    while (( N_REMAIN > 0 && ATTEMPTS < 6 * TOTAL_CHUNKS + 10 )); do
        ATTEMPTS=$(( ATTEMPTS + 1 ))
        N=$CHUNK
        (( N > N_REMAIN )) && N=$(( N_REMAIN ))
        CS="$CURSOR"
        CE=$(( CURSOR + N / 4 ))
        CURSOR="$CE"
        BEFORE=$(ls "$GAMES_DIR" | wc -l)
        echo ">>> [尝试 ${ATTEMPTS}] 块 ${N} 局 seed=[${CS},${CE}) $(date '+%F %T')"
        if ! run_chunk "$N" "$CS" "$GAMES_DIR" "$RUN_LOG"; then
            echo "!!! 块失败 seed=[${CS},${CE}) —— 换后续窗口重试" >&2
            FAILED+=("[${CS},${CE})")
            continue
        fi
        AFTER=$(ls "$GAMES_DIR" | wc -l)
        GOT=$(( AFTER - BEFORE ))
        if (( GOT != N )); then
            echo "ERROR: 块退出码 0 但只落盘 ${GOT}/${N} 局(seed=[${CS},${CE}))" >&2
            exit 1
        fi
        N_REMAIN=$(( N_REMAIN - GOT ))
        USED+=("[${CS},${CE})")
        echo ">>> 块 OK +${GOT} 局,累计 $(( GAMES - N_REMAIN ))/${GAMES} $(date '+%F %T')"
    done
    DONE_GAMES=$(( GAMES - N_REMAIN ))
    {
        echo "# akochan 趋势尺 seed 窗口记录(串行) ($(date '+%F %T'))"
        echo "tag=at-${TAG} games=${GAMES} chunk=${CHUNK} key=0x$(printf '%x' "$KEY_DEC")"
        echo "成功窗口(${DONE_GAMES}/${GAMES} 局):"
        printf '  %s\n' "${USED[@]:-}"
        echo "失败窗口(akochan 崩溃,已跳过):"
        printf '  %s\n' "${FAILED[@]:-}"
    } > "$WINDOWS"
    if (( DONE_GAMES < GAMES )); then
        echo "ERROR: 只完成 ${DONE_GAMES}/${GAMES} 局,尝试次数用尽(${ATTEMPTS});失败窗口: ${FAILED[*]:-none}" >&2
        echo "ERROR: 趋势不可用(akochan 反复崩溃),详见 $RUN_LOG / $WINDOWS" >&2
        exit 1
    fi
else
    # 并行:把 TOTAL_CHUNKS 块分给 PAR 个 worker(尽量均分),每个 worker 独立
    # seed 区间(SEED_START + k*AKO_SEED_STRIDE 起)与子目录;结束后合并。
    declare -a TARGETS=() WPIDS=()
    BASE_CHUNKS=$(( TOTAL_CHUNKS / PAR ))
    REM=$(( TOTAL_CHUNKS % PAR ))
    for ((k = 0; k < PAR; k++)); do
        CK=$BASE_CHUNKS
        (( k < REM )) && CK=$(( CK + 1 ))
        TARGETS[k]=$(( CK * 4 ))
    done
    : > "$RUN_LOG"
    for ((k = 0; k < PAR; k++)); do
        if (( TARGETS[k] == 0 )); then continue; fi
        WDIR="$GAMES_DIR/.w${k}"
        WSEED=$(( SEED_START + k * SEED_STRIDE ))
        echo ">>> [worker ${k}] target=${TARGETS[k]} 局 seed 起点=${WSEED} dir=${WDIR}"
        run_worker "$k" "${TARGETS[k]}" "$WSEED" "$WDIR" "$RUN_LOG.w${k}" &
        WPIDS+=("$!")
    done
    FAILED_PAR=0
    for p in "${WPIDS[@]}"; do
        wait "$p" || FAILED_PAR=1
    done
    # 合并子目录 + 汇总窗口/日志
    : > "$WINDOWS"
    MERGED=0
    for ((k = 0; k < PAR; k++)); do
        WDIR="$GAMES_DIR/.w${k}"
        if [[ -d "$WDIR" ]]; then
            CNT=$(ls "$WDIR"/*.json.gz 2>/dev/null | wc -l)
            MERGED=$(( MERGED + CNT ))
            mv "$WDIR"/*.json.gz "$GAMES_DIR"/ 2>/dev/null || true
            rmdir "$WDIR" 2>/dev/null || true
        fi
        if [[ -f "$RUN_LOG.w${k}" ]]; then
            cat "$RUN_LOG.w${k}" >> "$RUN_LOG"
            rm -f "$RUN_LOG.w${k}"
        fi
        if [[ -f "$RUN_LOG.w${k}.windows" ]]; then
            cat "$RUN_LOG.w${k}.windows" >> "$WINDOWS"
            rm -f "$RUN_LOG.w${k}.windows"
        fi
    done
    {
        echo "# akochan 趋势尺 seed 窗口记录(并行 ${PAR}) ($(date '+%F %T'))"
        echo "tag=at-${TAG} games=${GAMES} chunk=${CHUNK} parallel=${PAR} key=0x$(printf '%x' "$KEY_DEC")"
        echo "合并局数=${MERGED}/${GAMES}"
    } >> "$WINDOWS"
    if (( MERGED != GAMES )); then
        echo "ERROR: 合并后 ${MERGED}/${GAMES} 局(worker 未凑满目标或落盘异常),失败标记=${FAILED_PAR}" >&2
        echo "ERROR: 趋势不可用(akochan 反复崩溃),详见 $RUN_LOG* / $WINDOWS" >&2
        exit 1
    fi
fi

echo ">>> akochan 基准评估结束: $(date '+%F %T') | 完成 ${GAMES} 局 | 日志: $RUN_LOG"
echo ">>> 解析: $RUN_LOG"
python3 "$REPO_ROOT/scripts/parse_eval.py" "$RUN_LOG" --akochan | tee "$REPORT"
echo ">>> 报告: $REPORT"
echo ">>> 窗口记录: $WINDOWS"
