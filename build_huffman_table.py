#!/usr/bin/env python3
"""从字频 CSV 构建七叉哈夫曼码表, 生成 huffman_table.py(收发共用)。

用法:
    python3 build_huffman_table.py [字频CSV]

输入: 25 亿字语料字符频率表(14976 个简体常用字, 纯汉字, 无空格/标点/ASCII),
列: 序号, 字符, token 总次数, 频率(每百万), 累计覆盖率。本脚本只用"字符"和
"token"两列, 排序即语料频率排序。

合成权重: 语料不含空格/字母/数字/标点, 但真实消息离不开它们, 故追加一小块
按典型聊天文本比例人为设定的权重(见 SYNTHETIC, 可调整后重新生成)。
控制符号: ESCAPE(表外字符 -> UTF-8 字节逐字节编码)与 EOF(消息结束,
使码流自定界)的权重按语料总量比例给出。

七叉哈夫曼: 叶子数须满足 (n-1) % (K-1) == 0, 不足补 freq=0 哑叶;
哑叶不进入输出码表(码表因此是不完全前缀码, 前缀无歧义性不受影响)。
码字按规范(canonical)编号: 同长码字按字母表顺序连续编号,
    first[L+1] = (first[L] + cnt[L]) * K
生成模块只存"码长 -> 规范序符号串", 编解码两端用同一算法重建码字。
"""
from __future__ import annotations

import csv
import heapq
import sys
from collections import Counter
from fractions import Fraction

K = 7                                        # 信道数字字母表: 0..6
ESC_CHAR, EOF_CHAR = "\ue000", "\ue001"      # 控制符号占位(私用区)

DEFAULT_CSV = ("Chinese-character-list-from-2.5-billion-words-corpus-"
               "ordered-by-frequency.csv")

# --- 合成权重(语料缺失的真实消息常用符号; 依据典型聊天文本的相对频率估计) ---
SYNTHETIC = [
    (" ", 1.0e8), ("\n", 1.5e7),
    # 全角标点
    ("，", 4.0e7), ("。", 3.0e7), ("？", 6.0e6), ("！", 6.0e6), ("、", 8.0e6),
    ("：", 3.0e6), ("；", 1.5e6), ("…", 2.0e6), ("—", 1.5e6),
    ("“", 8.0e5), ("”", 8.0e5), ("‘", 4.0e5), ("’", 4.0e5),
    ("（", 5.0e5), ("）", 5.0e5), ("《", 5.0e5), ("》", 5.0e5),
    ("【", 3.0e5), ("】", 3.0e5), ("～", 3.0e5), ("·", 3.0e5),
    # 小写字母(英文自然频率的大致比例)
    ("e", 1.0e7), ("t", 7.0e6), ("a", 6.5e6), ("o", 6.0e6), ("i", 6.0e6),
    ("n", 6.0e6), ("s", 5.5e6), ("h", 5.0e6), ("r", 5.0e6), ("d", 4.0e6),
    ("l", 4.0e6), ("u", 3.5e6), ("c", 3.0e6), ("m", 3.0e6), ("w", 2.5e6),
    ("f", 2.5e6), ("g", 2.5e6), ("y", 2.5e6), ("p", 2.0e6), ("b", 1.8e6),
    ("v", 1.0e6), ("k", 1.0e6), ("j", 5.0e5), ("x", 5.0e5), ("q", 4.0e5),
    ("z", 4.0e5),
    # 大写字母
    *[(c, 4.0e5) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    # 数字
    ("0", 1.5e6), ("1", 1.2e6), ("2", 1.0e6), ("3", 9.0e5), ("4", 8.0e5),
    ("5", 8.0e5), ("6", 8.0e5), ("7", 7.0e5), ("8", 7.0e5), ("9", 7.0e5),
    # 半角标点
    (".", 3.0e6), (",", 3.0e6), ("?", 2.0e6), ("!", 2.0e6), ("'", 1.0e6),
    ('"', 8.0e5), (":", 8.0e5), ("-", 8.0e5), ("/", 5.0e5), ("(", 5.0e5),
    (")", 5.0e5), ("@", 4.0e5), ("#", 3.0e5), ("%", 3.0e5), (";", 3.0e5),
    ("+", 2.0e5), ("=", 2.0e5), ("_", 2.0e5), ("*", 2.0e5), ("&", 2.0e5),
]


def build(csv_path: str):
    rows = list(csv.reader(open(csv_path, encoding="utf-8")))
    hanzi = [(r[1], int(r[2])) for r in rows[1:] if len(r[1]) == 1]
    assert len(hanzi) == len({c for c, _ in hanzi}), "语料表内有重复字符"

    alphabet = hanzi + [(c, int(w)) for c, w in SYNTHETIC]
    total = sum(w for _, w in alphabet)
    alphabet += [(ESC_CHAR, total // 256), (EOF_CHAR, total // 16)]

    # --- 七叉哈夫曼(哑叶补齐, 保证每次合并恰好取 K 个) ---
    n_leaves = len(alphabet)
    dummies = (-(n_leaves - 1)) % (K - 1)
    weights = [w for _, w in alphabet] + [0] * dummies
    nodes: list = [None] * (n_leaves + dummies)   # 叶子; 内部节点存子节点表
    heap = [(w, i) for i, w in enumerate(weights)]
    heapq.heapify(heap)
    while len(heap) > 1:
        children = [heapq.heappop(heap)[1] for _ in range(K)]
        w = sum(weights[c] for c in children)
        nodes.append(children)
        weights.append(w)
        heapq.heappush(heap, (w, len(nodes) - 1))
    assert len(heap) == 1

    # --- 遍历求码长 ---
    lengths: dict[int, int] = {}
    stack = [(len(nodes) - 1, 0)]
    while stack:
        node, depth = stack.pop()
        if nodes[node] is None:
            lengths[node] = depth
        else:
            for child in nodes[node]:
                stack.append((child, depth + 1))

    # 完整性: 含哑叶的满 K 叉树, Kraft 和恒等于 1
    assert sum(Fraction(1, K ** L) for L in lengths.values()) == 1

    # --- 丢弃哑叶, 规范编号: 按 (码长, 字母表序) 分组连续编号 ---
    kept = {i: lengths[i] for i in range(n_leaves)}
    order = sorted(kept, key=lambda i: (kept[i], i))
    cnt = Counter(kept.values())
    codebook: list[tuple[int, str]] = []
    prev_end = 0
    for L in range(1, max(kept.values()) + 1):
        start = prev_end * K
        syms = "".join(alphabet[i][0] for i in order if kept[i] == L)
        assert len(syms) == cnt.get(L, 0)
        assert start + len(syms) <= K ** L, "规范编号溢出"
        if syms:
            codebook.append((L, syms))
        prev_end = start + len(syms)

    # 信息熵(数据符号, 供对照理论下限)
    prob = [w for _, w in alphabet]
    tot = sum(prob)
    entropy = sum(-p / tot * __import__("math").log2(p / tot) for p in prob if p)
    return codebook, entropy, n_leaves


HEADER = '''"""七叉哈夫曼规范码表 —— 收发共用(单一事实来源)。

由 build_huffman_table.py 从字频数据生成, 勿手改:
    字频: {csv} (25 亿字语料, 14976 个简体字, 纯汉字)
    另含少量合成权重的常用符号(空格/换行/字母/数字/标点, 见生成脚本
    SYNTHETIC 块)与两个控制符号: "\\ue000"=ESCAPE, "\\ue001"=EOF。

码字为 0..6 的数字序列; 存储形式是 (码长 L, 规范序符号串):
同长码字按串内顺序连续编号, 首码为上一码长结束处的"进位":
    code(L, 第 j 个) = first[L] + j,  first[1] = 0,
    first[L+1] = (first[L] + 该码长符号数) * 7
两端用同一算法重建码字, 解码器按 (已读长度, 数值) 前缀匹配。
"""
from __future__ import annotations

# (码长, 该码长的全部符号, 按规范顺序)
CODEBOOK: tuple[tuple[int, str], ...] = (
{body},
)
'''


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    codebook, entropy, n_syms = build(csv_path)
    body = ",\n".join(f"    ({L}, {s!r})" for L, s in codebook)
    with open("huffman_table.py", "w", encoding="utf-8") as f:
        f.write(HEADER.format(csv=csv_path, body=body))
    lens = [L for L, _ in codebook]
    print(f"生成 huffman_table.py: {n_syms} 个符号, 码长 {lens[0]}..{lens[-1]}, "
          f"数据符号熵 {entropy:.2f} bit/字符 "
          f"(理论下限 {entropy / 2.807:.2f} 数字/字符)")


if __name__ == "__main__":
    main()
