# 邮件分类项目文档

## 核心功能
基于朴素贝叶斯分类器实现垃圾邮件检测，支持高频词和 TF-IDF 两种特征模式，包含样本平衡和分类评估功能。

## 算法基础
- **多项式朴素贝叶斯**：假设特征条件独立，基于贝叶斯定理计算后验概率：
  \[
  P(y|x_1, x_2, ..., x_n) \propto P(y) \prod_{i=1}^n P(x_i|y)
  \]
- **应用场景**：通过词频或 TF-IDF 值构建特征，预测邮件类别。

## 数据处理流程
1. **分词**：使用 `jieba` 对文本精确分词。
2. **停用词过滤**：移除常见虚词（如“的”、“是”）。
3. **特征构建**：
   - **高频词**：选择词频 Top-N 的词。
   - **TF-IDF**：计算词频-逆文档频率加权值。

## 特征模式切换
在 `extract_features` 函数中指定参数 `mode`：
```python
# 使用高频词
X, vectorizer = extract_features(corpus, mode='frequency')

# 使用 TF-IDF
X, vectorizer = extract_features(corpus, mode='tfidf')