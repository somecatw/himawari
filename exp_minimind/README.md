# exp_minimind —— LM 信源编码实验

PLAN.md「LM 长文本档」的可行性实验: 用 MiniMind-3 (68.8M) 的逐位置
logit softmax 建七叉哈夫曼树做信源编码。**权重不入库**(gitignored),
按下面步骤取数复现。

## 文件

- `exp.py` — 字符级逐位置建树 + numpy 前向移植(无 torch 读 .pth) +
  token 级 CE 评测; 结论: 字符粒度上限 ~3.2 数字/字。
- `exp_tokens.py` — **token 级逐位置建树**(正式方案的原型): 压缩 1.87
  数字/字(-45%), 往返校验、码长分布、单数字删除损伤探针。

## 复现

```bash
# 依赖: numpy (仓库 requirements 已有) + tokenizers (pip, ~3MB, 无 torch)
pip install tokenizers

# 1) 权重 (ModelScope, 各 137MB, 放本目录)
curl -L -o pretrain_768.pth "https://modelscope.cn/models/gongjy/minimind-3-pytorch/resolve/master/pretrain_768.pth"
curl -L -o full_sft_768.pth "https://modelscope.cn/models/gongjy/minimind-3-pytorch/resolve/master/full_sft_768.pth"

# 2) tokenizer (minimind 仓库的 model/tokenizer.json; 位置用环境变量指)
git clone --depth 1 https://github.com/jingyaogong/minimind /tmp/minimind
export MINIMIND_TOKENIZER=/tmp/minimind/model/tokenizer.json

# 3) 跑 (CPU 单线程; 权重路径/样例见各脚本 main)
python3 exp.py          # 字符级 + 两个 checkpoint 对比
python3 exp_tokens.py   # token 级: 压缩率/往返/损伤探针
```

注意: `full_sft` 是对话微调权重, 裸文本建模用 `pretrain`; 实验脚本为
一次性原型, 正式模块见 PLAN.md T1。
