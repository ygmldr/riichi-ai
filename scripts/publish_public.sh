#!/usr/bin/env bash
# 把本地分支发布到公开 GitHub 仓库：重写历史（剔除 .omo/、替换邮箱）→ 校验 → force-push。
#
# 为什么需要重写：本地历史里有 .omo/ 会话笔记（decisions/learnings/evidence）和真实邮箱，
# 直接 push 会把它们公开。本脚本每次从本地仓库重新 clone 一份，只在克隆里过滤；
# 原仓库、当前 worktree 和 tenhou-integration 等分支都不受影响。
#
# 用法:
#   bash scripts/publish_public.sh              # 构建 + 校验 + 交互确认后 force-push
#   bash scripts/publish_public.sh --dry-run    # 只构建 + 校验，不 push
#   bash scripts/publish_public.sh --yes        # 跳过确认（明确授权时）
#   bash scripts/publish_public.sh --clean      # 成功后删除暂存克隆（保留备份）
#
# 环境变量:
#   PUBLISH_REMOTE    默认 git@github.com:ygmldr/riichi-ai.git
#   PUBLISH_BRANCH    默认 main（本地与远端同名）
#   PUBLISH_EMAIL     默认 ygmldr@users.noreply.github.com（写入作者与提交者）
#   PUBLISH_OLD_EMAIL 默认 1548999469@qq.com（被替换掉的邮箱）
#
# 产物:
#   .publish/backups/pre-publish-<ts>.bundle   全量 refs 备份（每次发布前）
#   .publish/repo/                             过滤后的暂存克隆（--clean 才删）
#
# 退出码: 0 成功; 1 前置条件/校验失败; 2 用户取消。
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${PUBLISH_REMOTE:-git@github.com:ygmldr/riichi-ai.git}"
BRANCH="${PUBLISH_BRANCH:-main}"
NEW_EMAIL="${PUBLISH_EMAIL:-ygmldr@users.noreply.github.com}"
OLD_EMAIL="${PUBLISH_OLD_EMAIL:-1548999469@qq.com}"

PUBLISH_ROOT="$REPO/.publish"
BACKUP_DIR="$PUBLISH_ROOT/backups"
STAGE="$PUBLISH_ROOT/repo"

DRY_RUN=0
ASSUME_YES=0
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --yes|-y)  ASSUME_YES=1 ;;
    --clean)   CLEAN=1 ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $arg (try --help)" >&2; exit 1 ;;
  esac
done

die()  { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARN:  $*" >&2; }
say()  { printf '\n== %s\n' "$*"; }

say "前置检查"
[[ -d "$REPO/.git" ]] || die "not a git repository: $REPO"
cd "$REPO"
git rev-parse --verify --quiet "$BRANCH^{commit}" >/dev/null || die "branch not found: $BRANCH"

# 远端必须已存在（空仓库 ls-remote 返回空列表、退出码 0；不存在的仓库会报错）。
if ! git ls-remote "$REMOTE" >/dev/null 2>&1; then
  die "无法访问远端 $REMOTE —— 请先在 GitHub 建一个空仓库（不要初始化 README/license）"
fi

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  warn "工作区有未提交的已跟踪改动；clone 只取已提交状态，这些改动不会被发布"
fi

# 暂存路径固定且必须位于本仓库内，避免 rm -rf 打偏。
case "$STAGE" in
  "$REPO"/.publish/repo) ;;
  *) die "unexpected staging path: $STAGE" ;;
esac

say "备份全部 refs"
mkdir -p "$BACKUP_DIR"
BUNDLE="$BACKUP_DIR/pre-publish-$(date +%Y%m%d-%H%M%S).bundle"
git bundle create "$BUNDLE" --all >/dev/null
git bundle verify "$BUNDLE" >/dev/null
echo "bundle: $BUNDLE ($(du -h "$BUNDLE" | cut -f1))"

say "重建暂存克隆"
rm -rf -- "$STAGE"
git clone -q --no-hardlinks "$REPO" "$STAGE"
cd "$STAGE"
git checkout -q "$BRANCH"

SRC_HEAD="$(git -C "$REPO" rev-parse --short "$BRANCH")"
SRC_COUNT="$(git rev-list --count "$BRANCH")"
SRC_FILES="$(git -C "$REPO" ls-tree -r --name-only "$BRANCH" | wc -l)"

say "重写历史（删除 .omo/，邮箱 $OLD_EMAIL -> $NEW_EMAIL）"
export OLD_EMAIL NEW_EMAIL
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
  --index-filter 'git rm -r --cached --ignore-unmatch .omo' \
  --env-filter '
    if [ "$GIT_AUTHOR_EMAIL" = "$OLD_EMAIL" ]; then export GIT_AUTHOR_EMAIL="$NEW_EMAIL"; fi
    if [ "$GIT_COMMITTER_EMAIL" = "$OLD_EMAIL" ]; then export GIT_COMMITTER_EMAIL="$NEW_EMAIL"; fi
  ' \
  --prune-empty -- "$BRANCH" >/dev/null

# 丢掉 filter-branch 的备份 ref 与 clone 带进来的远端跟踪 ref，再清对象。
git symbolic-ref -d refs/remotes/origin/HEAD 2>/dev/null || true
git for-each-ref --format='delete %(refname)' refs/original | git update-ref --stdin || true
git for-each-ref --format='delete %(refname)' refs/remotes  | git update-ref --stdin || true
git reflog expire --expire=now --all
git gc -q --prune=now

say "校验"
FAIL=0
check() { if eval "$2"; then echo "  ok   $1"; else echo "  FAIL $1"; FAIL=1; fi; }

check "历史中无 .omo/"        '[[ $(git log --all --oneline -- .omo | wc -l) -eq 0 ]]'
check "工作树中无 .omo/"      '[[ $(git ls-files | grep -c "^\.omo") -eq 0 ]]'
check "不含权重 (*.pth)"      '! git ls-tree -r --name-only HEAD | grep -qE "(^|/)\.pth$|^checkpoints/"'
check "作者/提交者邮箱唯一"    '[[ $(git log --all --format="%ae%n%ce" | sort -u | wc -l) -eq 1 ]]'
check "邮箱为 $NEW_EMAIL"      '[[ $(git log --all --format="%ae" | sort -u) == "$NEW_EMAIL" ]]'
check "无 >1MB 的文件"        '[[ $(git ls-tree -r -l HEAD | awk "\$4 > 1048576" | wc -l) -eq 0 ]]'
check "树内容仅少了 .omo/"     '! diff <(git ls-tree -r --name-only HEAD) <(git -C "$REPO" ls-tree -r --name-only "$BRANCH" 2>/dev/null) | grep "^[<>]" | grep -qv "\.omo/"'

if [[ $FAIL -ne 0 ]]; then
  die "校验未通过，未 push（暂存克隆保留在 $STAGE）"
fi

DST_HEAD="$(git rev-parse --short HEAD)"
DST_COUNT="$(git rev-list --count HEAD)"
DST_FILES="$(git ls-tree -r --name-only HEAD | wc -l)"

cat <<EOF

== 发布摘要
  源:      $SRC_HEAD  $SRC_COUNT commits  $SRC_FILES files
  发布:    $DST_HEAD  $DST_COUNT commits  $DST_FILES files
  远端:    $REMOTE  (branch $BRANCH)
  最大文件: $(git ls-tree -r -l HEAD | sort -k4 -n | tail -1 | awk '{print $4" B  "$5}')
EOF

if [[ $DRY_RUN -eq 1 ]]; then
  echo
  echo "dry-run: 未 push；暂存克隆在 $STAGE"
  exit 0
fi

REMOTE_SHA="$(git ls-remote "$REMOTE" "refs/heads/$BRANCH" | cut -f1 || true)"
if [[ -n "$REMOTE_SHA" ]]; then
  warn "远端 $BRANCH 已存在 ($(git rev-parse --short "$REMOTE_SHA" 2>/dev/null || echo "$REMOTE_SHA"))，本次会 force-push 覆盖"
fi

if [[ $ASSUME_YES -eq 0 ]]; then
  echo
  read -r -p "输入 yes 执行 force-push: " answer
  [[ "$answer" == "yes" ]] || { echo "已取消"; exit 2; }
fi

say "push"
git remote remove gh 2>/dev/null || true
git remote add gh "$REMOTE"
timeout 300 git push --force gh "HEAD:refs/heads/$BRANCH" 2>&1 | tail -3

say "推送后核验"
git fetch -q gh
[[ "$(git ls-remote gh "refs/heads/$BRANCH" | cut -f1)" == "$(git rev-parse HEAD)" ]] \
  || die "远端 HEAD 与本地不一致"
check "远端树中无 .omo/" '[[ $(git ls-tree -r --name-only "gh/$BRANCH" | grep -c "^\.omo") -eq 0 ]]'
check "远端树中无权重"    '! git ls-tree -r --name-only "gh/$BRANCH" | grep -qE "(^|/)\.pth$|^checkpoints/"'
[[ $FAIL -eq 0 ]] || die "远端核验失败，请检查 $REMOTE"

URL="${REMOTE%.git}"
case "$URL" in
  git@github.com:*) URL="https://github.com/${URL#git@github.com:}" ;;
  git@*)            URL="https://${URL#git@}" ;;
  https://*)        ;;
  *)                URL="(无法从 $REMOTE 推导网页地址)" ;;
esac
echo
echo "完成: $URL  ($DST_HEAD, branch $BRANCH)"
echo "备份: $BUNDLE"

if [[ $CLEAN -eq 1 ]]; then
  rm -rf -- "$STAGE"
  echo "已删除暂存克隆 $STAGE（备份保留）"
fi
