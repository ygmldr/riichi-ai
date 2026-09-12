# 配置

- `config.example.toml` — 模板(路径指向 /root/riichi-ai)。训练前复制为 `config.toml` 并调整:
  - `[grp.dataset]` 的 train_globs / val_globs(按年份切分)
  - `[1v3.champion]` / `[1v3.challenger]` 的 state_file(评估对象;challenger 默认指向训练产物 mortal.pth,champion 指向 baseline.pth)
  - `[1v3]` 的 iters(每个 iter = games_per_iter 局;首次验证先保持 1)
  - `[1v3.akochan]`(可选对手,获取/构建见 RUNBOOK「准备 akochan 对手」)
- 运行时通过环境变量指定:MORTAL_CFG=/root/riichi-ai/configs/config.toml
- 训练脚本须在 vendor/mortal/mortal/ 目录下运行(相对导入)
