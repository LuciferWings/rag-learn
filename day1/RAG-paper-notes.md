RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

# model RAG-Sequence vs. RAG-Token:
### Langchain 默认支持 RAG-Sequence 风格
### HuggingFace transformers 两个都支持 RagTokenForGeneration， RagSequenceForGeneration

| 特性       | RAG-Sequence              | RAG-Token              |
| -------- | ------------------------- | ---------------------- |
| 生成粒度     | 基于整个 sequence             | 基于每个 token             |
| 条件依赖     | 每个文档独立生成完整答案              | 每个 token 融合所有文档        |
| 最终输出选择方式 | 从各文档生成的 sequence 中选择概率最高的 | 所有文档生成 token 的分布做加权平均  |
| 训练时 loss | 对每个文档独立计算 loss，再求和        | 融合所有文档的 log-likelihood |


| 应用类型                  | 推荐方式                      |
| --------------------- | ------------------------- |
| 快速原型，API 交互，LangChain | Prompt 拼接（stuff, refine）  |
| 精准问答，科研，自主部署模型        | Token-level 融合（RAG-Token） |
| 模型推理不透明（如 GPT）        | 只能用 Prompt 拼接             |
| 可控生成（哪些文档影响哪些词）       | 推荐 RAG-Token              |


# Sparse Retrieval vs. Dense Retrieval:

| 类型                         | 描述                                | 示例                              | 特点              |
| -------------------------- | --------------------------------- | ------------------------------- | --------------- |
| **稀疏检索（Sparse Retrieval）** | 使用词频或关键词索引，向量维度极高但大多数为 0          | **BM25**, TF-IDF                | 快速、高效、但语义理解弱    |
| **密集检索（Dense Retrieval）**  | 使用神经网络将 query 和文档编码为低维向量（如 768-d） | **DPR**, BGE, OpenAI Embeddings | 更能理解语义关系，支持语义搜索 |


# BLEU and ROUGE
- BLEU (Bilingual Evaluation Understudy) 是用来评估 生成文本（如回答）和参考文本 的相似程度的指标，广泛用于机器翻译和文本生成任务。BLEU-1 只考虑 1-gram（即单个词）重合情况。分数范围是 0～1，越高表示生成结果越接近参考答案。
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation) 是另一种文本评估指标，常用于总结、问答等任务中。ROUGE-L 特别关注 最长公共子序列（LCS），强调词序关系。

| 应用类型                  | 推荐方式                      |
| --------------------- | ------------------------- |
| 快速原型，API 交互，LangChain | Prompt 拼接（stuff, refine）  |
| 精准问答，科研，自主部署模型        | Token-level 融合（RAG-Token） |
| 模型推理不透明（如 GPT）        | 只能用 Prompt 拼接             |
| 可控生成（哪些文档影响哪些词）       | 推荐 RAG-Token              |


# traditional sparse vector TF-IDF，BM25(改进的TF-IDF)
Lucene 默认使用 BM25
TF-IDF(t,d,D)=TF(t,d)×IDF(t,D)
1. Term Frequency (TF)
Measures how frequently a term appears in a document.

2. Inverse Document Frequency (IDF)
Measures how common or rare a term is across all documents.



# Dense Retrieval 相似度计算公式

在稠密检索（Dense Retrieval）中，常用的相似度计算公式为：

$$
\text{sim}(q, p) = E_Q(q)^T E_P(p)
$$

- $q$：查询（query）
- $p$：文档（passage）
- $E_Q(q)$：查询的向量表示（通常由编码器得到）
- $E_P(p)$：文档的向量表示

该公式表示查询向量与文档向量的点积（内积），衡量它们在向量空间中的相似度。点积越大，表示两者越相似。

# 双编码器 可以一样可以不一样
| 特性   | 共享编码器     | 不共享编码器    |
| ---- | --------- | --------- |
| 参数量  | 较小        | 较大        |
| 输入类型 | 类型相同      | 类型可不同     |
| 表达能力 | 可能弱一些     | 更灵活       |
| 适用任务 | 句子匹配、检索等  | 多模态、上下文任务 |
| 举例   | DPR、SBERT | CLIP、对话系统 |

# 相似度计算
点积，余弦，欧式距离

# Contrastive Learning with SimCLR
Family of self-supervised larning methods
4096-8192

# 我的理解:
- TF-IDF属于稀疏向量空间模型，而RAG用的稠密向量模型。是为了解决什么问题呢？
当我们回答一些像近词义的词时，如果用TF-IDF可能就会查询失败，因为TF-IDF是基于词频的，
- 稀疏向量空间模型是高纬度的，稠密向量是低纬度的

