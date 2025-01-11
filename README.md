# uncertainty quantification

This repository is created for archiving the result of 2024 UROP research scholarship program. 

## Code Reference

https://arxiv.org/abs/2402.03744

https://github.com/alibaba/eigenscore

## 1. Background

Despite the remarkable performance of large language models (LLMs) across various NLP tasks, the issue of hallucination—producing inaccurate or false answers—remains a critical concern. The main problem with hallucination is that it generates fluent and plausible but incorrect responses, which can mislead users and ultimately reduce trust and reliance on LLMs. Uncertainty Estimation is a research field that measures the degree of hallucination in LLM responses and distinguishes between accurate and false answers, playing a crucial role in building safe and trustworthy LLMs. This study proposes two approaches for uncertainty estimation using the internal states of LLMs, based on prior research (Chen et al., 2024). First, it utilizes the scaled-dot product used in self-attention calculations. Second, it employs cosine similarity, a metric for measuring the similarity between vectors. The superior performance of these two methods was validated using two off-the-shelf LLaMA models and two question-answering benchmarks.

## 2. Method

This study proposes two uncertainty estimation approaches using the internal states of LLMs, inspired by the findings from INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection (Chen et al., 2024). Since the hidden states of LLMs are represented as d-dimensional vector embeddings, we hypothesized that metrics for measuring similarity between text vector embeddings can be used to quantify consistency across outputs. This study quantifies similarity using scaled-dot product and cosine similarity.

### 1. Attention Score

The attention map in Transformers represents relationships between tokens in a sequence and can be computed using the scaled-dot product formula shown in Eq. (4). The core of this formula, softmax((QK^T)/√(d_k)), serves as a weighting factor for V, indicating how strongly the i-th token in the sequence is related to the j-th token (Vaswani et al., 2017).

Recognizing that the attention map reflects semantic associations between tokens in a sequence, we hypothesized that modifying the scaled-dot product formula could measure consistency across multiple outputs. Instead of transforming hidden states into matrices Q and K, we directly compute the inner product of the hidden states and scale them by √(d_k). Notably, we exclude the softmax function, as applying it would obscure the distinction between high and low consistency. In the standard scaled-dot product, softmax normalizes the relative attention weights among tokens (j = 1, 2, ..., K). However, for consistency measurement, absolute attention is more relevant.

For example, consider the question, "Who is the author of?" and assume the model generates the outputs "Hermann Hesse," "Hermann Hesse," and "William Shakespeare." By computing the inner product of the concatenated matrix of their hidden states and scaling by √(d_k), a 3x3 matrix is obtained. If the values in the second and third rows are [1.2, 2.0, 0.3] and [0.2, 0.3, 1.0], respectively, applying softmax would yield [0.2753, 0.6127, 0.1119] and [0.2309, 0.2552, 0.5139]. In this case, the second row (corresponding to "Hermann Hesse") exhibits high semantic similarity to other outputs, indicating it is likely correct. Conversely, the third row (corresponding to "William Shakespeare") shows low similarity, suggesting semantic inconsistency. However, softmax normalization transforms these absolute similarities into relative values, diluting the difference in consistency between "Hermann Hesse" and "William Shakespeare." Therefore, we modify the scaled-dot product calculation by removing the softmax function, as shown in Eq. (5). The concatenated matrix H, composed of K hidden states, has dimensions K x d_k, and the attention map computed in Eq. (5) has dimensions K x K.

<div align="center">
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/9f6c3b50-20b6-4d66-85a8-5f556b517e90" />
  <p>Eq. (4)</p>
</div>
<div align="center">
  <img width="400" alt="image" src="https://github.com/user-attachments/assets/9f6c3b50-20b6-4d66-85a8-5f556b517e90" />
  <p>Eq. (5)</p>
</div>

The attention score is calculated using Eq. (6), which averages the values of the upper triangular matrix, excluding the diagonal. Since the hidden states are directly used, the upper and lower triangular matrices are symmetric, and the diagonal represents the self-inner product, which is excluded. A higher attention score indicates greater consistency across outputs.

<div align="center">
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/aff2aec5-217c-4593-9459-ea32ab52e29a" /> 
  <p>Eq. (6)</p>
</div>

### 2. Cosine Score

Cosine similarity measures the similarity between two vector embeddings based on the cosine of the angle between them. It ranges from -1 to 1, where values closer to 1 indicate more similar directions. Since hidden states are embedding vectors, cosine similarity can quantify the similarity (or consistency) across outputs. Using Eq. (7), cosine similarity is calculated for (K(K-1))/2 pairs of hidden states, and the average cosine similarity across all pairs is computed using Eq. (8) to obtain the Cosine score. A higher Cosine score indicates greater consistency across outputs.

<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/dc8cda53-b879-48b5-b675-58d644a057bb" />
  <p>Eq. (7)</p>
</div>
<div align="center">
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/0340ea85-1efb-4489-8b79-b9028a58b3b5" /> 
  <p>Eq. (8)a</p>
</div>

Figure 2 illustrates the overall pipeline for computing Attention score and Cosine score:

The LLM is given a query, and K outputs are generated. In the experiments, K was set to 10.
A concatenated matrix of K hidden states is constructed, analogous to an embedding matrix for a sequence of K words.

### Extraction of Hidden States

Previous studies have shown that the hidden states from the intermediate layers of a model best preserve the semantic information of the output (Chen et al., 2024). As illustrated in Figure 1, the AUROC is higher when using the last token of the intermediate layer's hidden states compared to the first or last layer. Based on these findings, this study uses the last token of the intermediate layer's hidden states without conducting separate experiments.

<div align="center">
  <img width="300" alt="image" src="https://github.com/user-attachments/assets/3f97ac26-e777-4928-a4bc-d6aa509cef3c" /> 
  <p>Figure 1 (Chen, et al., 2024)</p>
</div>

The process involves:

Using the concatenated matrix to calculate the Attention score and Cosine score.
Classifying outputs as correct if the scores are high and as incorrect if the scores are low.

<div align="center">
  <img width="800" alt="image" src="https://github.com/user-attachments/assets/6c33faa3-2cb4-4f12-bfd4-b866b3c188b7" />
  <p>Figure 2: Pipeline of similarity computation</p>
</div>

## 3. Experiments

### Dataset and Model 

I used SQuAD and triviaqa for dataset and LLaMA-7B and LLaMA-13B for model. 

### Correctness Metric and Evaluation Metric

Correctness metric is a metric that determines whether model's generation is correct or not. I labeled the most likely generation of the model either 1 (correct) or 0 (incorrect), computing Rouge-L score between ground-truth answer and the model's generation. Evaluation metric is a metric that evaluates the performance of the uncertainty estimation approach. I used AUROC which is useful for evaluating the performance of binary classifier. The higher AUROC means the better performance of the model. (Ren, et al., 2022; Lin, et al., 2023)

### Baseline

I used perplexity, Length-Normalized Entropy, Lexical Similarity as baselines. 

## 4. Result 

### Effectiveness of Attention Score and Cosine Score in Hallucination Detection

Performance in LLaMA-7B and SQuAD dataset: Figure 4 (a)
The AUROCs of the Attention score and Cosine score were 0.79 and 0.791, respectively, showing similar performance. The three hidden states-based metrics—Eigenscore, Attention score, and Cosine score—outperformed all other n-gram-based (Lexical Similarity) or probability-based (Perplexity, LN-Entropy) metrics in terms of AUROC. At the optimal threshold calculated using the G-Mean, the accuracies of Eigenscore and Attention score were the highest, at 0.73 and 0.727, respectively.

Performance in LLaMA-7B and TriviaQA dataset: Figure 4 (b)
The ground truth of the TriviaQA dataset consists of short answers (mostly one or two words), making n-gram-based and probability-based metrics perform best. However, the three hidden states-based metrics still demonstrated decent performance. The Attention score achieved an AUROC of 0.786, while the Cosine score (0.82) was comparable to the Eigenscore (0.827).

Performance in LLaMA-13B and SQuAD dataset: Figure 5 (a)
While Eigenscore achieved the highest AUROC (0.831), the Attention score (0.802) and Cosine score (0.808) also outperformed n-gram-based and probability-based metrics. The accuracies at the optimal threshold calculated using the G-Mean were the highest for Eigenscore (0.741) and Attention score (0.721).

Performance in LLaMA-13B and TriviaQA dataset: Figure 5 (b)
Lexical Similarity and Perplexity achieved the highest AUROCs, 0.844 and 0.841, respectively. Although the Attention score recorded a relatively lower AUROC of 0.772, the Cosine score achieved a stable and high AUROC of 0.818, showcasing strong hallucination detection capabilities.

The Pearson Correlation Coefficient (PCC) was consistently high for both Eigenscore and Cosine score but remained close to zero for Attention score across all four combinations. This discrepancy arises because, in the LLaMA-7B and SQuAD dataset, the optimal thresholds for Eigenscore and Cosine score were 1.405 and 0.5907, respectively, while the Attention score's optimal threshold was significantly larger at 19.9375. As a result, metrics like Rouge-L, Eigenscore, and Cosine score, which have values in the range of [0, 1], showed strong correlations, while Attention score did not. This pattern also appeared in LLaMA-13B, where the optimal thresholds for Attention score were much higher: 96.875 for SQuAD and 67.8125 for TriviaQA, indicating that the Attention score varies greatly depending on the model and dataset.

These experiments demonstrate that hidden states, which contain semantic information beyond lexical overlap, perform exceptionally well in uncertainty estimation. Although the Attention score was nearly uncorrelated with correctness metrics, it consistently showed high AUROCs and accuracies. Additionally, the Cosine score demonstrated consistently high AUROCs along with strong PCC values, proving its superiority and generalizability as a hallucination detection metric. Table 2 summarizes the overall results.

### Advantage of Hidden States Metrics: Superior Hallucination Detection in Long Responses in Real-World Scenarios

Although the hidden states-based metrics performed slightly worse than n-gram-based and probability-based metrics on the TriviaQA dataset, it is important to note that the Attention score and Cosine score achieved significantly high AUROCs on the SQuAD dataset. In practice, LLM outputs are rarely as short as those in TriviaQA. The ground truth of the SQuAD dataset contains many full sentences, making it a better reflection of real-world LLM outputs compared to TriviaQA. On the SQuAD dataset, the performance of Eigenscore, Attention score, and Cosine score was consistently superior on both LLaMA-7B and LLaMA-13B. Thus, this experiment suggests that leveraging hidden states in real-world uncertainty estimation is more effective than relying on lexical overlap or generation probabilities.

<div align="center">
  <img width="619" alt="image" src="https://github.com/user-attachments/assets/05dd754b-8d56-4f5e-9fdd-7cc3ada25a0d" /> 
  <p>Figure 4 (a): LLaMA-7B and SQuAD</p>
</div>

<div align="center">
  <img width="637" alt="image" src="https://github.com/user-attachments/assets/31143b63-1fdb-4776-b14a-57690c2d0182" /> 
  <p>Figure 4 (b): LLaMA-7B and triviaqa</p>
</div>

<div align="center">
  <img width="626" alt="image" src="https://github.com/user-attachments/assets/abff1222-d3e4-42e9-9915-60dd47f2c149" />
  <p>Figure 5 (a): LLaMA-13B and SQuAD</p>
</div>

<div align="center">
  <img width="624" alt="image" src="https://github.com/user-attachments/assets/d24be29a-4ec0-49ef-9dc0-c89437c83461" /> 
  <p>Figure 5 (b): LLaMA-13B and triviaqa</p>
</div>

<div align="center">
  <img width="873" alt="image" src="https://github.com/user-attachments/assets/80e424c6-ce76-4c60-86b4-1c2395a3fbab" />
  <p>Table 2:〖AUC〗_r: AUROC when using the correctness metric with Rouge-L > 0.5. PCC: Pearson Correlation Coefficient between the correctness metric and the hallucination detection metric. ACC: Accuracy of the hallucination detection metric at the optimal threshold defined by G-Mean </p>
</div>

## Conclusion 

This study proposed two hallucination detection approaches utilizing the hidden states of LLMs: the Attention Score and the Cosine Score, and demonstrated their performance and utility. First, the Attention Score approach was introduced based on the fact that the scaled-dot product used in the transformer’s attention mechanism computes semantic associations between tokens in a sequence. Second, the Cosine Score was proposed by leveraging the simple fact that hidden states are text vector embeddings and calculating the similarity between text embeddings using cosine similarity. While the Attention Score showed variability depending on the model and dataset, it exhibited excellent hallucination detection performance. The Cosine Score consistently achieved the second-highest AUROC and PCC after the Eigenscore, regardless of the model or dataset. This study is significant in that it proposed uncertainty estimation approaches that are both interpretable in process and results, and practical in real-world applications by employing simple operations like the scaled-dot product and cosine similarity.

## Reference 

Chen, C., Liu, K., Chen, Z., Gu, Y., Wu, Y., Tao, M., Fu, Z., & Ye, J. (2024). INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection. ArXiv, abs/2402.03744.

Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation. ArXiv, abs/2302.09664.

Ren, J., Luo, J., Zhao, Y., Krishna, K., Saleh, M., Lakshminarayanan, B., & Liu, P.J. (2022). Out-of-Distribution Detection and Selective Generation for Conditional Language Models. ArXiv, abs/2209.15558.

Malinin, A., & Gales, M.J. (2021). Uncertainty Estimation in Autoregressive Structured Prediction. International Conference on Learning Representations.

Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E.H., & Zhou, D. (2022). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ArXiv, abs/2203.11171.

Manakul, P., Liusie, A., & Gales, M.J. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. ArXiv, abs/2303.08896.

Lin, C. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. Annual Meeting of the Association for Computational Linguistics.

Ji, Z., Chen, D., Ishii, E., Cahyawijaya, S., Bang, Y., Wilie, B., & Fung, P. (2024). LLM Internal States Reveal Hallucination Risk Faced With a Query. ArXiv, abs/2407.03282.

Duan, J., Cheng, H., Wang, S., Wang, C., Zavalny, A., Xu, R., Kailkhura, B., & Xu, K. (2023). Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models. ArXiv, abs/2307.01379.

Gao, X., Zhang, J., Mouatadid, L., & Das, K. (2024). SPUQ: Perturbation-Based Uncertainty Quantification for Large Language Models. ArXiv, abs/2403.02509.

Lin, S.C., Hilton, J., & Evans, O. (2022). Teaching Models to Express Their Uncertainty in Words. Trans. Mach. Learn. Res., 2022.

Vaswani, A., Shazeer, N.M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). Attention is All you Need. Neural Information Processing Systems.

Lin, Z., Trivedi, S., & Sun, J. (2023). Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models. Trans. Mach. Learn. Res., 2024.















