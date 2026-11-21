# 运行手册(P0-P3)

## 双机拓扑
- 控制台(/root/riichi-ai): 开发、数据下载、评估结果解析
- 训练机(5070Ti): git pull + 环境 + 训练

## 控制台一次操作
```bash
cd /root/riichi-ai
python3 scripts/download_data.py datasets   # 12.78GB zip,断点续传
python3 scripts/verify_data.py datasets     # 校验 + 统计局数
```

## 数据上机(控制台 → 5070Ti)
`datasets/` 下的年度 zip(合计 12.78GB)不进 git,需手动传输到 5070Ti。

传输示例(任选其一,rsync 支持断点续传,推荐):
```bash
# 控制台上
rsync -avP --partial datasets/ user@5070ti:/root/riichi-ai/datasets/
# 或 SFTP
sftp user@5070ti
#   sftp> put -r datasets/  (或逐年度 put datasets/20XX.zip)
```

5070Ti 磁盘需求:
- zip 合计: 12.78GB
- 解压后 .mjson 约 40-60GB(`prepare_data_on_gpu.sh` 解压到 `datasets/extracted/`)
- 加上仓库、venv、torch 等,建议 5070Ti 可用磁盘 ≥ 100GB

解压并校验通过后可按需删除 zip 省空间(控制台有存档,随时可重传):
```bash
rm datasets/*.zip
```

## 5070Ti 一次操作
```bash
git clone --recurse-submodules <repo-url> riichi-ai
cd riichi-ai
bash scripts/setup_gpu_env.sh               # Python3.12 venv + torch cu128 + libriichi
bash scripts/prepare_data_on_gpu.sh datasets # 解压 + GRP globs
cp configs/config.example.toml configs/config.toml
# 编辑 config.toml: [grp.dataset] globs、[dataset] globs、[control] state_file 等
bash scripts/run_grp.sh configs/config.toml   # P2: GRP 训练
bash scripts/run_train.sh configs/config.toml # P3: 离线 DQN 训练
bash scripts/run_eval.sh configs/config.toml  # P3 验收: 1v3 duplicate mahjong
```

注: 4 个 runner 脚本(grp/train/resume/eval)都会自动 `mkdir -p checkpoints`,无需手工创建。

数据格式注意: 2025/2026 年度的 zip 内 `.mjson` 是**未压缩明文 JSON**(其余年度为 gzip),
而 Mortal 加载器(libriichi `load_gz_log_files`)严格要求 gzip。`prepare_data_on_gpu.sh` 已内置
`scripts/normalize_gzip.py` 会自动规范化;若手动解压数据,务必自己跑一遍该脚本,否则训练会在
随机读到相关文件时崩溃(`invalid gzip header`,2026-09-14 实际发生过)。

## baseline.pth 引导(P3 DQN 训练前置)
`train.py` 启动时无条件构造 `TestPlayer()`(vendor/mortal/mortal/train.py:97),立即加载
config 的 `[baseline.test] state_file`(`checkpoints/baseline.pth`)。该文件缺失时
`run_train.sh` 会打印错误并以非 0 退出(不会静默创建空文件)。

首次训练前,先在 vendor/mortal/mortal/ 下生成随机初始化引导 baseline:
```bash
cd vendor/mortal/mortal
MORTAL_CFG=/root/riichi-ai/configs/config.toml python3 - <<'EOF'
import torch
from config import config
from model import Brain, DQN
v = config['control']['version']
torch.save({'config': config,
            'mortal': Brain(version=v, **config['resnet']).eval().state_dict(),
            'current_dqn': DQN(version=v).eval().state_dict()},
           config['baseline']['train']['state_file'])
EOF
```

首轮训练完成后,用真实检查点更新 baseline(后续每轮均可这样迭代):
```bash
cp checkpoints/mortal.pth checkpoints/baseline.pth
```

注意: 首轮 test_play 对比的是随机基线,提升判定基本不触发,首轮不产出 `best.pth` 属
**预期行为**,不影响 `mortal.pth` 产出。

## 监控与续训
- GRP 日志: runs/grp_train.log
- DQN 日志: runs/dqn_train.log
- 续训: 把 config.toml 的 state_file 指向 checkpoints/mortal.pth, 然后 resume_train.sh
- TensorBoard: tensorboard --logdir runs

## 1v3 评估(P3 验收)
运行: `bash scripts/run_eval.sh configs/config.toml`,产物为 `runs/1v3/eval_<时间戳>.log`。

config.toml 的 [1v3] 段编辑清单:
- `[1v3.challenger] state_file` → `checkpoints/mortal.pth`(训练产物,模板已指向)
- `[1v3.champion] state_file` → `checkpoints/baseline.pth`(见上节 baseline 引导)
- `[1v3] iters`: 每个 iter 跑 `games_per_iter` 局(默认 2000);模板默认 iters = 1 用于
  首次验证(几分钟),验收时再加大(注意 iters = 500 即 100 万局,近似无限跑)
- `[1v3.akochan] enabled = false` 时对手为 baseline 模型,无需 akochan 即可验收

### akochan 对手(2026-09-16 已构建,见下方踩坑记录)

已就绪:`vendor/akochan/` 含 `system.exe`(二进制)、`libai.so`、`tactics.json`、`params/`、
`.boost_local/`(本地 Boost 前缀)。启用只需把 config.toml 的 `[1v3.akochan].enabled` 置 true
(路径默认已对);里程碑评估用 `bash scripts/eval_akochan.sh` 会自动处理。

重建时的踩坑记录(**非 Rust 项目,README 里曾误写 cargo**):
1. 源码: `git clone https://github.com/critter-mj/akochan.git vendor/akochan`(C++/GNU Make)
2. Boost 依赖:系统无 boost 且无 sudo → 用 `apt-get download` 下载 `libboost1.83-dev`、
   `libboost-system1.83-dev`、`libboost-system1.83.0`,再用 `dpkg-deb -x` 解到
   `vendor/akochan/.boost_local`(约 190MB,免 root)
3. 两段构建(libai.so → system.exe);system.exe 的 RUNPATH 必须是 `$ORIGIN`,
   否则 Mortal spawn 时找不到 libai.so:
   ```bash
   cd vendor/akochan/ai_src && nice -n 10 make -f Makefile_Linux -j4 \
     LIBS="-I../.boost_local/usr/include -L../.boost_local/usr/lib/x86_64-linux-gnu -lboost_system"
   cd .. && nice -n 10 make -f Makefile_Linux -j4 \
     LIBS="-I.boost_local/usr/include -L.boost_local/usr/lib/x86_64-linux-gnu -lboost_system -I. -L. -lai -Wl,-rpath,\$$ORIGIN"
   ```
4. tactics.json 必须是 `{"tactics": {...}}` 包装格式(akochan `set_tactics_one` 断言该键);
   内容取自 Equim-chan/mjai-reviewer 的同名文件(jun_pt [90,45,0,-135],与 strength.md 官方跑分一致)
5. 接线(libriichi `src/agent/akochan.rs`):以 `cwd=vendor/akochan` 执行
   `system.exe pipe <tactics> <player_id>`;akochan 用相对路径读 `params/`,故 cwd 必须正确

akochan 非必需: 用我们自己的训练产物(baseline/旧快照)作 champion 同样满足验收链路。

## 验收标准(P3 完成)
解析评估日志,challenger 相对 champion 的均顺位分 > 0 即通过:
```bash
python3 scripts/parse_eval.py runs/1v3/eval_*.log
```
(对手为 baseline 或 akochan 均可;parse_eval.py 用法详见其 docstring)

## 里程碑评估(akochan 绝对基准)

每 **50 万步**(0.5M/1M/1.5M/2M/2.5M/3M…)执行一次 akochan 基准,用于绝对参照
(与上游 `vendor/mortal/docs/src/perf/strength.md` 的数字直接可比):

```bash
bash scripts/eval_akochan.sh          # 10000 局(默认;2500 sets)
bash scripts/eval_akochan.sh 4000     # 快速版(噪声约 ±1.7)
```

脚本行为:快照 `checkpoints/mortal.pth` → `runs/akochan_eval/mortal.pth.at-<steps>` →
生成临时配置(`[1v3.akochan].enabled=true`、challenger=快照、固定 seed_key) →
`run_eval.sh` + `parse_eval.py --akochan`(与训练**并发**运行,训练会变慢)。

**速度预期(实测 2026-09-16)**:akochan 每决策约 0.4-1.1s(前几次含参数加载需数秒),一局里
akochan 一个席位约 100+ 次决策 → **1 set(4 局,4 局并行)实测 ≈ 13.5 分钟**(训练并发时)。
因此:**40 局 ≈ 2.3h;100 局 ≈ 5.6h;400 局 ≈ 22h;2000 局 ≈ 4-5 天**。
- 里程碑建议 `games=40~100` 做趋势观察(噪声约 ±5~8pt);要严谨绝对值需后台跑数百局以上
- 上游 strength.md 的 akochan 对照表用了 1.3-7 万局(在本机这个速度不现实),故我们只做趋势
- 注意:WSL 下 SIGSTOP 无法可靠暂停训练(实测进程仍为 R 状态),所以评估只与训练并发跑;
  训练期间评估会让训练慢 ~10-15%

解读:**日志中 "challenger" 是 akochan(x1),我方模型坐 3 个 champion 席位**;
"我方每席位均分 > 0"即强于 akochan。上游参照:akochan 每席位 -5.67pt(Mortal 3.1 时代),
v4 约 -8~-9pt(推算)。
