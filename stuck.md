# nanoGPT 盲写记录

## Day 3 热身（Bigram model）

### Q1: 字符级 tokenizer 要做什么？
统计每个字符出现的次数，每个字符前面的字符是什么

### Q2: Bigram 模型的核心是一个什么形状的 embedding table？
二维矩阵

### Q3: 训练时的 batch 长什么样？shape 是什么？
一个batch应该是一个矩阵，假设一个batch要训练n条数据，统计每个字符和前面三个字符，那应该是一个n行4列矩阵

### Q4: 从模型 forward 出来的 logits shape 是什么？loss 怎么算？
是一个二维矩阵，loss可以用cross-entropy算,或者计算logits的矩阵和原矩阵每个元素对应的均方误差

### 修正后

Q1: tokenizer = 字符 ↔ 整数 id 的双向字典（stoi, itos），加 encode/decode 函数。不涉及统计。

Q2: embedding table 形状 (V, V)。V = vocab_size。
    行 i = "看到字符 i 时，下一个字符的 logits"。

Q3: batch 是 (xb, yb)，shape 都是 (B, T)。
    xb[i, t] = 位置 t 的字符，yb[i, t] = 位置 t 的下一个字符。
    T 不是上下文长度，是"每条序列并行提供 T 个训练位置"。

Q4: logits shape (B, T, V)。用 cross-entropy。
    算之前要把 logits reshape 成 (B*T, V)，yb 成 (B*T,)。


## 好问题 #1 (Day 3)
Q: 为什么 Shakespeare bigram 不加 BOS/EOS token？
A: 流式字符级 LM，没有句子边界；`\n` 已自然扮演软边界。
   现代大模型（GPT-2 的 <|endoftext|>, LLaMA 的 <s>/</s>）会加，
   在 seq2seq / 对话 / 多文档混合预训练时必要。

## 好问题 #2 (Day 3, Phase 3.2)
Q: idx 和 targets 分别是什么？
A: idx (B, T) 是输入字符 id；targets (B, T) 是"每个位置的下一个字符" id (ground truth)。
   模型在 T 个位置并行预测，每个位置独立出一个 (V,) 的 logits，
   loss = 这 T 个位置 cross-entropy 的平均。
   targets=None 时跳过 loss 计算（用于生成）。
