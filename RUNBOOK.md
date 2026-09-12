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

### 准备 akochan 对手(可选)
想对更强对手验收时:
1. 获取源码(任一):
   ```bash
   git clone https://github.com/critter-mj/akochan.git vendor/akochan
   ```
   或从已有环境拷贝已构建的 akochan 目录(需含二进制与 tactics.json)。
2. 构建(需 Rust 工具链):
   ```bash
   cd vendor/akochan
   cargo build --release
   ```
3. 启用: config.toml 中 `[1v3.akochan] enabled = true`(dir/tactics 默认已指向
   `vendor/akochan/`,如 clone 到别处则改路径)。
4. 重新运行 `bash scripts/run_eval.sh configs/config.toml`。

akochan 非必需: 用我们自己的训练产物(baseline 模型)作 champion 同样满足验收链路。

## 验收标准(P3 完成)
解析评估日志,challenger 相对 champion 的均顺位分 > 0 即通过:
```bash
python3 scripts/parse_eval.py runs/1v3/eval_*.log
```
(对手为 baseline 或 akochan 均可;parse_eval.py 用法详见其 docstring)
