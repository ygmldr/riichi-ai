# vendor/

`mortal/` 是 Equim-chan/Mortal 的 git 子模块(fork 基准)。

- 固定 commit:`0cff2b52982be5b1163aa9a62fb01f03ce91e0d2`(见 `git -C vendor/mortal log --oneline -1`,2026-08-09 固定)
- 同步上游:`git submodule update --remote vendor/mortal`(需审慎,先跑测试)
- 约定:不修改 `vendor/mortal/mortal/` 下任何代码;自定义配置通过 `MORTAL_CFG` 环境变量注入;确需代码改动时以 `patches/` 补丁文件记录、按需应用
- 2026-09-13 重建:原仓库 git 历史丢失后,按上述 pin 重建(浅克隆 `--depth 1`);`target/` 构建产物(libriichi.so)已保留复用

## 本地补丁

- `patches/0001-train-grp-dataloader-num-workers-0.patch` — `mortal/train_grp.py` 的 DataLoader `num_workers` 1→0(2026-08-10 GRP 实机训练所用)。重训 GRP 前应用:
  ```bash
  git -C vendor/mortal apply "$(pwd)/patches/0001-train-grp-dataloader-num-workers-0.patch"
  ```
