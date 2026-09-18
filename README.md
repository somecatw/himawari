# himawari —— 游戏口琴信道通信系统

把游戏里的口琴乐器当作数据信道: 发送端把文本编成音符序列, 玩家照着弹;
接收端采集游戏音频, 从音高还原符号流, 解码出文本。

```
文本 ──编码器──> 音符序列 ──人演奏──> 游戏音频 ──音高识别(frontend/)──> 符号流 ──解码──> 文本
       notes_codec / huffman_codec                engine.py + symbols.py
```

## 信道约定

- 乐器音域 C3~C6, 当前使用 **C4~B4 共 7 个数据音** + C5;
- 硬约束: **不允许两个连续同音**(人无法用换气区分它们);
- 数据符号即 GF(7) 元素 0..6, 与音符的对应关系由 `frontend/symbols.py` 的
  `CodecSpec` 定义(单一事实来源, 键位 zxcvbnm + `,`)。

## 发送端编码器

| 文件 | 方案 |
|---|---|
| `notes_codec.py` | baseline: UTF-8 每字节固定展开 3 个基-7 数字(大端), 相邻同音插 C5 转义 |
| `huffman_codec.py` | 当前: 七叉哈夫曼(按字频) + **旋转信道层** |
| `build_huffman_table.py` | 从 25 亿字语料字频表生成码表 `huffman_table.py` |
| `huffman_table.py` | 规范码表(15082 符号, 码长 2~11), 收发共用, 勿手改 |

旋转信道层: 把 8 个音(C4..C5)看成一个环, 数字 d 编为「上一音符 + 1 + d (mod 8)」,
相邻音符天然不同, **8 个音全部承载数据**——相对 C5 转义方案省掉 1/7 的转义税,
平均码长恰为其 **7/8**(转义率 = 相邻同数字概率 ≈ 1/7)。

用法:

```bash
python3 huffman_codec.py encode "你好, world"   # 文本 -> 音名序列
python3 huffman_codec.py encode "hi" --midi     # MIDI 编号; --keys 输出键位见 notes_codec
python3 huffman_codec.py decode "D4 F4 ..."     # 音名序列 -> 文本
python3 huffman_codec.py stats                  # 与 baseline 的码长对比
python3 huffman_codec.py selftest
```

## 实验

### 1. 旋转信道层 vs C5 转义

旋转层的音符数恒等于数字数(无转义), 转义方案期望多 (n-1)/7 个音符,
实测与 7/8 理论值吻合(见 `huffman_codec.py stats` 输出的两列音符数之比)。

### 2. 压缩方案对比

统一折算成「信道上要吹的音符数」。ts_zip 的字节数 × 3(按 baseline 的
3 数字/字节分帧; 理论最优 base-7 打包为 ×2.85, 不影响结论)。

| 样例 | 字节 | gzip | xz | baseline 音符 | 哈夫曼音符 | ts_zip 字节 | ts_zip 折算音符 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ok | 2 | 22 | 60 | 6 | **9** | 15 | 45 |
| 在吗？(3 字) | 9 | 30 | 68 | 31 | **12** | 22 | 66 |
| 你好，世界！(6 字) | 18 | 41 | 76 | 60 | **21** | 23 | 69 |
| Hello, world! | 13 | 33 | 72 | 44 | 44 | 21 | **63** |
| 今晚八点老地方见(13 字) | 39 | 62 | 96 | 136 | **46** | 37 | 111 |
| emoji + 生僻字(19 字) | 40 | 63 | 96 | 143 | 92 | 39 | **117** |
| 49 字中文 | 121 | 144 | 184 | 423 | 180 | 55 | **165** |
| 187 字公告 | 537 | 440 | 512 | 1874 | 693 | 153 | **459** |

结论:

1. **聊天尺度的消息(< 100 字节)哈夫曼完胜**。ts_zip 有 15~23 字节固定
   格式开销, gzip/xz 纯头部负收益, 都不适合短消息。
2. **交叉点约 100~120 字节(约 35~40 汉字)**, 此后 LLM 压缩优势随长度扩大:
   187 字公告 ts_zip 折算 459 音符 vs 哈夫曼 693 (1.5 倍)。
3. 187 字公告 ts_zip 折算 **2.45 音符/字**, 已低于字频表的 order-0 熵下限
   3.25 数字/字(`build_huffman_table.py` 输出 9.12 bit/字)——差距来自预测
   模型(词级 BPE + 上下文)对字频模型的降维打击, 这就是路线 3 的天花板价值。

### 3. ts_zip 实验环境

- 官方 ts_zip 2024-03-02 + RWKV 169M v4 (int8, `rwkv_169M_q8.bin` 171MB);
- CPU 模式(x86_64, 4 线程): 105 B/s, 537 字节约 5 s(不含模型加载);
- 压缩/解压往返校验一致;
- 实用性硬伤: 双端需同一 171MB 模型且推理逐位可复现, 换模型版本/硬件
  可能解压失败(官方标注 experimental)。作为实际信道不现实, 作为码长
  天花板参照很有价值。
- 二进制与模型不入库(见 .gitignore), 本地放在 `ts_zip-2024-03-02/`。

复现:

```bash
cd ts_zip-2024-03-02 && ./ts_zip c msg.txt msg.ts && ./ts_zip d msg.ts msg.out
python3 huffman_codec.py stats
```

## 状态与下一步

- [x] baseline 编码器 (notes_codec.py)
- [x] 七叉哈夫曼 + 旋转信道层 (huffman_codec.py), 流式协议默认不发 EOF
- [x] 接收端: 流式贪心解码 + 单错误枚举 (receiver.py), GUI (frontend/gui.py)
- [x] RS 纠错三档 no/medium/high + 停顿分帧 (rs_codec.py)
- [x] LM 信源编码可行性实验: token 级逐位置建树 1.87 数字/字 (-45%), 见 `exp_minimind/`
- [ ] **LM 长文本档产品化**: 任务分解与招募见 [PLAN.md](PLAN.md)
- [ ] order-1 静态表蒸馏(用 SFT 后模型导出 P(c|h), 零运行时依赖的中间档)
