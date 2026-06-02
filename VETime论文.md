# VETime: Vision Enhanced Zero-Shot Time Series Anomaly Detection

Yingyuan Yang 1 Tian Lan 1 Yifei Gao 1 Yimeng Lu 1 Wenjun He 2 Meng Wang 2 Chenghao Liu 3 Chen Zhang 1 

# Abstract

Time-series anomaly detection (TSAD) requires identifying both immediate Point Anomalies and long-range Context Anomalies. However, existing foundation models face a fundamental tradeoff: 1D temporal models provide fine-grained pointwise localization but lack a global contextual perspective, while 2D vision-based models capture global patterns but suffer from information bottlenecks due to a lack of temporal alignment and coarse-grained pointwise detection. To resolve this dilemma, we propose VETime, the first TSAD framework that unifies temporal and visual modalities through fine-grained visual-temporal alignment and dynamic fusion. VETime introduces a Reversible Image Conversion and a Patch-Level Temporal Alignment module to establish a shared visual-temporal timeline, preserving discriminative details while maintaining temporal sensitivity. Furthermore, we design an Anomaly Window Contrastive Learning mechanism and a Task-Adaptive Multi-Modal Fusion to adaptively integrate the complementary perceptual strengths of both modalities. Extensive experiments demonstrate that VETime significantly outperforms state-of-the-art models in zero-shot scenarios, achieving superior localization precision with lower computational overhead than current vision-based approaches. Code available at: https://github.com/yyyangcoder/VETime. 

# 1. Introduction

Time-Series Anomaly Detection (TSAD) is a fundamental yet challenging problem, which requires identifying rare, subtle, and often non-stationary deviations while providing precise temporal localization. In practice, the heterogeneity 1Department of Industrial Engineering, Tsinghua University, Beijing, China 22012 Lab, Huawei Technologies Ltd, Beijing, China 3Datadog AI Research. Correspondence to: Chenghao Liu <twinsken@gmail.com>, Chen Zhang <chenzhang01@tsinghua.edu.cn>. 

Preprint. February 19, 2026. 

of time-series domains and deployment scenarios renders dataset-specific training impractical, while many real-world settings operate in low-resource or cold-start regimes where collecting data for reliable model training is infeasible. This drives the development of Time-Series Foundation Models (TSFMs) that support zero-shot anomaly detection across diverse distributions. 

However, a robust general anomaly detector must simultaneously tackle two distinct patterns: Point Anomalies, which manifest as abrupt, instantaneous numerical deviations (Shentu et al., 2024b); and Context Anomalies, characterized by large-scale contiguous irregularities in trend or periodicity (He et al., 2025), as shown in Figure 1 (a). Current uni-modal foundation models face a fundamental dilemma, showing a critical competency gap in effectively handling both anomaly types simultaneously, as shown in Figure 1 (b). 

Constrained by the intrinsic nature of the one-dimensional (1D) temporal perspective, current time series-based TSFMs operate with limited receptive fields confined to narrow local windows. While this design allows them to excel at the fine-grained localization of point anomalies by capturing local numerical continuity, it creates a bias against modeling long-range dependencies (Das et al., 2024; Goswami et al., 2024). Consequently, they lack the macroscopic perspective required to identify context anomalies. Conversely, vision-based approaches (Wu et al., 2023; Chen et al., 2025) seek to transform 1D sequences into 2D visual formats to capture high-level correlations from a global view. While these holistic representations(Xu et al., 2025; Zhou & Yu, 2024) successfully capture context anomalies, they are fundamentally constrained by the need to map variable-length sequences into fixed-size image inputs (e.g., 224 × 224) of standard vision backbones. The resulting information bottleneck causes visual blurring of raw signals, leading to inevitable over-detection with coarse-grained anomaly windows that significantly exceed the precise localization of the actual outliers. 

A natural progression to resolve this dilemma is to integrate these modalities to exploit their combined strengths. While Time-VLM (Zhong et al., 2025) represents the pioneering attempt to bridge time-series and vision-language models, its design is inherently tailored for forecasting tasks. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/04f0bddc8edfcb6cc04e63cc823409641b600861607e78e825da369c930994d3.jpg)



(a) Anomaly Definitions


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/82f125f1b784e3a2040830082c7180ffdf8bc5d90d154d9844e50387c4b1a92a.jpg)



(b) The Unimodal Dilemma


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/28b7ad143e8e42be528f3adea9f66b21ad436f274e7d938e5c89f58c1be4b9b8.jpg)



(c) The Proposed Solution



Figure 1. Comparison of the previous TSAD methods and the proposed VETime.


Anomaly detection presents a unique set of challenges compared to forecasting: rather than predicting global trends, it requires the precise identification of deviations at specific timestamps. This creates two distinct obstacles: Visual-Temporal Alignment Bottleneck. Visual features are deprived of explicit temporal coordinates as time-series-tovisual conversion obscure the intrinsic temporal indexing. To facilitate the fine-grained information interaction required for robust anomaly discrimination, it is imperative to effectively place both visual and temporal modalities onto a shared timeline. Dynamic Synergy for Diverse Anomaly Patterns. Given that the time-series modality offers superior local correlations while the image modality excels at capturing macroscopic patterns, the other critical challenge is how to adaptively fuse these complementary yet heterogeneous attributes and provide comprehensive information for effective anomaly detection. 

To bridge these gaps, we propose VETime, the first framework that unifies temporal and visual features via effective alignment and fusion, enabling robust anomaly detection and precise localization, as shown in Figure 1 (c). To overcome the alignment barrier, we first develop a novel Reversible Image Conversion method which constructs information-dense images with discriminative anomaly de-Long-term tails. Complementing this, we propose the Patch-Level Irregularity Temporal Alignment module, which enhances 2D visual representations from the pre-trained ViT with a 1D tem-Long-term poral ordering, establishing the structural basis for fine-Trendgrained cross-modal interaction. Subsequently, to facilitate dynamic synergy, we introduce two key mechanisms. Considering the distinct perceptual characteristics of visual and temporal modalities regarding anomalies, we propose a specialized Anomaly Window Contrastive Learning. This incorporates intra- and inter-window Contrast between multi-modal features to achieve comprehensive anomaly identification. Finally, Task-Adaptive Multi-Modal Fusion module equipped with a reconstruction head is applied to facilitate robust multi-modal fusion. This mechanism performs dynamic, adaptive weighted fusion of the aligned features, ensuring precise performance in downstream anomaly detection. Our main contributions are listed as follows: 

• We propose VETime, the first TSAD framework that integrates visual and temporal features by fine-grained alignment and dynamic fusion, effectively exploiting the distinct perceptual advantages of each modality. 

• We introduce an efficient Reversible Image Conversion module and a Patch-Level Temporal Alignment Alignmentmodule, which jointly capture information-rich visual contexts while preserving critical temporal sensitivities. 

• We introduce the Anomaly Window Contrastive Learning and Task-Adaptive Multi-Modal Fusion, synthesizing multi-modal complementary perceptual strengths to ensure comprehensive anomaly capture. 

• Extensive experiments on multiple TSAD datasets prove that the proposed framework significantly outperforms existing SOTA models with lower computational costs compared to current vision-based methods. 

# 2. Related Work

# 2.1. Methods for TSAD

Traditional TSAD algorithms primarily rely on reconstruction and forecasting paradigms (Hundman et al., 2018; Su et al., 2019; Tuli et al., 2022), they often struggle to generalize across diverse domains due to their training on specific datasets. To address these limitations, Time Series Foundation Models (Shentu et al., 2024a; Goswami et al., 2024; Ansari et al., 2024; Das et al., 2024; Lan et al., 2025) have emerged, attempting to capture universal patterns via large-scale pre-training. However, TSFMs often suffer from over-generalization, where powerful reconstruction capabilities inadvertently reconstruct anomalies as well as normal data, masking the discrepancy required for detection. A promising emerging direction involves visual approaches that leverage Vision-Language Models (VLMs) (Xu et al., 2025; Zhuang et al., 2024; He et al., 2025) to capture global context under macroscopic perspective. Nevertheless, most existing works overlook the intrinsic perceptual complementarity between raw 1D temporal precision and 2D global visual context, falling short of effectively detecting both point and context anomalies in a unified framework. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/b1f253dc30d0c5941080c9f67c6fd4c707d696c8ba97a61e9daf76bd708cc449.jpg)



Figure 2. Overview of the proposed framework. The time series is first processed by a time-series encoder to extract temporal features $F _ { T S }$ while simultaneously undergoing Reversible Image Conversion to generate visual input. Then visual features $F _ { V _ { 0 } }$ extracted from a frozen pre-trained image encoder are subsequently transformed into $F _ { V }$ through Patch-Level Temporal Alignment to reinforce their temporal positional association. Then, $F _ { T S }$ and $F _ { V }$ are input into the Anomaly Window Contrastive Learning to derive an anomaly-enhanced representation $F _ { A }$ . Finally, a Task-Adaptive Multi-Modal Fusion module integrates all features $( F _ { A } , \bar { F _ { T S } } , F _ { V } )$ . Final outputs $F _ { A D }$ and $F _ { R e c }$ are mapped to the original sequence length via token projection for the respective anomaly classification and reconstruction heads.


# 2.2. Vision-based Model for Time Series Analysis

Vision-based time series analysis leverages computer vision backbones by transforming 1D temporal data into 2D representations (Ni et al., 2025), mainly categorized into Line Plots (Liu & et al., 2025a;b), Heatmaps (Chen et al., 2025; Wang et al., 2024), and other transformation methods (Yang & et al., 2024; Ruan & Zhong, 2024). However, by treating the transformed visual representation as a holistic input, these pure vision-based approaches struggle to recover the precise sequential ordering of the original time series. Recently, Time-VLM (Zhong et al., 2025) attempts to bridge this gap by mapping raw time-series features into the vision-language model (VLM) feature space for multimodal fusion. Nevertheless, it still fails to establish a strict alignment between the transformed image features and the original temporal axis, limiting its ability to support the precise localization of anomalies required for effective TSAD. 

# 3. Methodology

# 3.1. Overview

To harness the differentiated yet complementary advantages of temporal and visual modalities, we propose a unified framework capable of fine-grained temporal alignment and dynamic interaction for anomaly detection. As illustrated in Figure 2, the framework comprises four core components: 

• Reversible Image Conversion: Transforms 1D time series into reversible visual representations. This module encapsulates both global and local correlations into semantically rich visual formats, thereby significantly accentuating anomalous patterns within the data. 

• Patch-Level Temporal Alignment: Realigns visual features to the temporal axis to ensure fine-grained semantic correspondence, establishing a robust prerequisite for the subsequent multi-scale interaction between global and local contexts. 

• Anomaly Window Contrastive Learning: Leverages the complementary nature of visual and temporal modalities to enforce multi-scale discriminability for anomaly information. Through intra- and inter-window contrastive learning, this module facilitates the mutual reinforcement of visual and temporal features to effectively isolate anomaly patterns. 

• Task-Adaptive Multi-Modal Fusion: Dynamically integrates multimodal features to achieve end-to-end 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/56fce51ed5c49a975a292f0c25ea2911c1c20c9fe521fc0c9fbab9856a515cb5.jpg)



Figure 3. The framework of Reversible Image Conversion and Patch-level Temporal Alignment


feature fusion. Specifically, it employs sequence reconstruction as an auxiliary constraint to promote deep feature interaction and comprehensive complementarity, ultimately generating robust anomaly representations. 

Ultimately, the fused representations capitalize on the synergy between temporal and visual domains, facilitating accurate localization across diverse anomaly types. It is noted that our framework supports univariate and multivariate time series settings. The main text focuses on univariate cases, with the multivariate extension detailed in Appendix A. 

# 3.2. Reversible Image Conversion:

To accentuate the distinctiveness of anomalous signals while maintaining efficient information encoding, we transform univariate time series into high-density visual representations via a three-stage pipeline, as shown in Figure 3 (a). 

Multi-Channel Intensity Mapping: Unlike prior singlechannel mapping approaches (Chen et al., 2025), we employ a multi-channel encoding strategy, which capitalizes on the standard RGB architecture to incorporate denser information and better exposes latent anomalous patterns. Following DLinear (Zeng et al., 2023), the raw series $X \in \mathbb { R } ^ { L }$ is decomposed into a trend component $X _ { t r e n d }$ and a remainder component $X _ { r e m } ,$ , where L is the length of the series. The triplet $\{ X , X _ { t r e n d } , X _ { r e m } \}$ is independently normalized to [0, 255] and mapped to the R, G, and B channels, yielding ${ \mathrm { ~ a ~ 1 ~ } } \times L \times 3$ tensor that explicitly encompasses both global trends and high-frequency residuals. 

Adaptive Folding: To accommodate diverse temporal scales, we adopt a periodicity-based folding strategy (Wu et al., 2023) to transform 1D sequence into a 2D grid with dimensions $( T _ { f o l d } , \lceil L / T _ { f o l d } \rceil , 3 )$ . The folding period $T _ { f o l d }$ is dynamically estimated via the autocorrelation function√ (defaulting to $\sqrt { L }$ for non-periodic samples), and adjusted to be a multiple of the ViT patch size to prevent temporal information discontinuity. For indivisible lengths, we use the mean of the final period to pad trailing segments, minimizing distribution shifts compared to zero-padding. 

Dimension-Aware Scaling: To standardize input resolution from $( T _ { f o l d } , \lceil L / T _ { f o l d } \rceil , 3 )$ to (224, 224, 3) without fidelity loss, we apply a decoupled scaling strategy. We use linear interpolation along the time axis (horizontal) to preserve waveform continuity, while employing a copypadding strategy along the period axis (vertical) to prevent semantic distortion or pseudo-patterns that could arise from interpolation across distinct periods. 

# 3.3. Patch-Level Temporal Alignment

The generated visual representations are fed into a frozen visual encoder to extract initial visual features $F _ { V _ { 0 } } \in$ $\mathbb { R } ^ { N _ { V } \times D _ { V } }$ . In parallel, the time series undergoes instance normalization and patching, followed by a temporal encoder to yield $F _ { T S } \in \mathbb { R } ^ { N _ { T S } \times D _ { T S } }$ . Here, N and D denote the patch count and feature dimension for the respective modalities. To bridge the structural discrepancy between modalities, we introduce a Patch-Level Temporal Alignment module for $F _ { V _ { 0 } }$ (Figure 3 (b)). 

Specifically, visual features are mapped back to the 1D temporal domain by inverting the folding logic. $F _ { V _ { 0 } }$ is reshaped to the initial 2D grid, linearly interpolated along the temporal axis to match the temporal patch count, and average-pooled along the periodicity axis to aggregate redundant repetitions. The resulting features are flattened to yield the concise aligned image features $\hat { F } _ { V } \in \mathbb { R } ^ { N _ { T S } \times D _ { V } }$ . 

Finally, to recover temporal context lost during visual encoding, we incorporate learnable positional encoding $E _ { P O S } \in$ $\mathbb { R } ^ { \breve { N } _ { T S } \times D _ { V } }$ followed by projection and self-attention layers to model intra- and inter-patch dependencies. Ultimately, this produces a visual representation $F _ { V }$ that maintains temporal correspondence with $F _ { T S }$ . 

# 3.4. Anomaly Window Contrastive Learning

Time-series and image features $F _ { T S }$ and $F _ { V }$ are first input into a dual cross-attention. Here, features from each modality serve as mutual queries to integrate complementary contexts, followed by a residual FFN for refinement, yielding updated representations $Z _ { T S }$ and $Z _ { V }$ . The process is formulated as: 

$$
Z _ {T S} ^ {0} = \operatorname{CrossAttn} (F _ {T S}, F _ {V}),
$$

$$
\begin{array}{l} Z _ {V} ^ {0} = \text { CrossAttn } (F _ {V}, F _ {T S}), \\ Z _ {V} ^ {0} = Z _ {V} ^ {0} + \text { EEN } (Z _ {V} ^ {0}) \end{array} \tag {1}
$$

$$
Z _ {T S} = Z _ {T S} ^ {0} + \operatorname{FFN} \left(Z _ {T S} ^ {0}\right),
$$

$$
Z _ {V} = Z _ {V} ^ {0} + \operatorname{FFN} \left(Z _ {V} ^ {0}\right),
$$

where 

$$
\operatorname{CrossAttn} (X, Y) = \operatorname{Softmax} \left(\frac {X W _ {Q} \left(Y W _ {K}\right) ^ {\top}}{\sqrt {d}}\right) \left(Y W _ {V}\right), \tag {2}
$$

denotes the cross-attention mechanism with X serving as queries and Y providing keys and values; $W _ { Q } , W _ { K }$ , and $\bar { W } _ { V } \in \mathbb { R } ^ { d \times d }$ are learnable projection matrices; d is the dimension of the feature embeddings; and FFN(·) represents a feed-forward network applied with residual connection. 

Given the inherent disparities between time-series and image features, specifically concerning their receptive fields and sensitivity to anomalies, we employ a hybrid strategy comprising Intra-Window and Inter-Window contrastive learning to explicitly model discriminative features across different scales, as shown in Figure 4. 

Anomaly Context Windows: To facilitate this, we construct adaptive windows around anomalies. Point-level labels are converted to binary patch-level labels (set to 1 if any timestamp within the patch is anomalous). Then, for a continuous anomaly segment spanning $L _ { a }$ patches, we define an Anomaly Context Window by symmetrically extending the normal segment on both sides, with a maximum length of $L _ { w } = 2 L _ { a }$ . Each window contains one continuous anomaly instance surrounded by its immediate local context. 

Intra-Window Contrastive Learning: Designed to capture short-duration point anomalies $( L _ { w } \leq 1$ patches, like outliers), this component enforces fine-grained anomaly alignment for visual feature $Z _ { V }$ . Within a window, the visual feature at the anomaly position serves as the anchor $Z _ { V } ^ { A }$ , paired with the corresponding temporal feature as the positive sample $Z _ { T S } ^ { A }$ . Normal temporal features within the same window act as negatives $Z _ { T S } ^ { N }$ . The intra-window contrastive loss $\mathcal { L } _ { i n t r a }$ is defined as: 

$$
\mathcal {L} _ {i n t r a} = - \log \frac {\exp (Z _ {V} ^ {A} \cdot Z _ {T S} ^ {A}) / \tau)}{\sum_ {k \in \mathcal {N} _ {n e g}} \exp (Z _ {V} ^ {A} \cdot Z _ {T S} ^ {k}) / \tau)}, \tag {3}
$$

where · denotes dot product similarity, τ is the temperature parameter, and $\mathcal { N } _ { n e g }$ represents the set of normal features within the window. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/6544b2b29fc5bdc4d6b849367f0ff622f84ab1c04173320077171fe8d167ea01.jpg)



Figure 4. An illustration of Anomaly Window Contrastive Learning


Inter-Window Contrastive Learning: For long-duration context anomalies $( L _ { w } \ge 2$ patches, like seasonal anomaly), we enhance global discriminability for temporal feature $Z _ { T S }$ . We aggregate features along the temporal dimension within each window using average pooling. The pooled time-series feature of an anomaly window $\hat { Z } _ { T S } ^ { \hat { A } }$ serves as the anchor, paired with its visual counterpart $\bar { Z } _ { V } ^ { A }$ , while pooled features from normal windows $\hat { Z } _ { V } ^ { N }$ serve as negatives.The inter-window contrastive loss $\mathcal { L } _ { i n t e r }$ is: 

$$
\mathcal {L} _ {\text { inter }} = - \log \frac {\exp (\bar {Z} _ {T S} ^ {A} \cdot \bar {Z} _ {V} ^ {A}) / \tau)}{\sum_ {j \in \mathcal {W} _ {\text { neg }}} \exp (\bar {Z} _ {T S} ^ {A} \cdot \bar {Z} _ {V} ^ {j}) / \tau)} \tag {4}
$$

where $\mathcal { W } _ { n e g }$ denotes the set of normal window features. 

Total contrastive Loss: The total contrastive loss $\mathcal { L } _ { a w }$ is: 

$$
\mathcal {L} _ {a w} = \frac {1}{N} \sum_ {i = 1} ^ {N} (\frac {1}{W _ {s}} \sum_ {s = 1} ^ {W _ {s}} \mathcal {L} _ {i n t r a} ^ {(s)} + \frac {1}{W _ {l}} \sum_ {l = 1} ^ {W _ {l}} \mathcal {L} _ {i n t e r} ^ {(l)}), \tag {5}
$$

where N is the number of samples, $W _ { s }$ and $W _ { l }$ are the number of short and long anomalies in each instance, respectively. Crucially, this loss is calculated independently for each sample. This design choice avoids interference caused by the distinct variability of normal and anomalous patterns across different instances, ensuring robust feature learning. 

Finally, the enhanced features are concatenated and processed by a Feed-Forward Network (FFN) to yield the anomaly representation $F _ { A } = \mathrm { F F N } \big ( [ Z _ { T S } ; Z _ { V } ] \big )$ , where [·; ·] denotes feature concatenation. 

# 3.5. Task-Adaptive Multi-Modal Fusion

To realize dynamic semantic integration and efficient endto-end training, we propose a Task-Adaptive Multi-modal Fusion Module. This module acts as a dynamic routing mechanism, treating time-series $( F _ { T S } )$ , vision $( F _ { V } )$ , and anomaly-enhanced $( F _ { A } )$ features as experts in a candidate space. 

Dynamic Weight Routing: A routing network computes patch-level dynamic weights $w \in \mathbb { R } ^ { N _ { T S } \times 3 \times 2 }$ to assign importance to each expert for the two downstream tasks (anomaly detection and reconstruction). Specifically, weights are generated via a learnable task-specific bias $\bar { \mathbf { b } _ { t a s k } } \in \mathbb { R } ^ { \bar { N } _ { T S } \times D _ { T S } \times 2 }$ added: 

$$
w = \operatorname{Softmax} \left(\mathrm{MLP} \left(F _ {A}\right) + \mathbf {b} _ {\text { task }}\right) \tag {6}
$$

where the Softmax is applied across the expert dimension. 

To prevent expert collapse where the router relies solely on a single modality, we impose an entropy-based regularization to encourage diverse expert utilization: 

$$
\mathcal {L} _ {e} = \frac {1}{2 N _ {T S}} \sum_ {i = 1} ^ {N _ {T S}} \sum_ {t = 1} ^ {2} \sum_ {m \in \{T S, V, A \}} w _ {i, m, t} \log w _ {i, m, t} \tag {7}
$$

where $N _ { T S }$ denotes the number of temporal patches, $t \in$ {1, 2} indexes the two downstream tasks including anomaly detection and reconstruction, m iterates over the three features, and $w _ { i , m , t }$ represents the dynamic fusion weight assigned to modality m for patch i in task t. 

Task-Adaptive Fusion: The final fused $F _ { F u s e d }$ is obtained by the weighted summation of the expert features: 

$$
F _ {F u s e d} = \sum_ {m \in \{T S, V, A \}} w _ {m} \cdot F _ {m} \tag {8}
$$

where $F _ { F u s e d } = [ F _ { A D } , F _ { R e c } ]$ is subsequently used for anomaly detection and reconstruction tasks. Crucially, we employ sequence reconstruction as an auxiliary task rather than a primary objective. By requiring the fused features to reconstruct the original input, we encourage the preservation of rich semantic content and foster deep multimodal interaction. This auxiliary constraint prevents the model from overfitting to sparse anomaly labels, ultimately yielding a more robust representation for the primary anomaly detection task. 

# 3.6. Optimization

The features $F _ { A D }$ and $F _ { R e c }$ are projected to the original sequence length and fed into separate specific heads. The model is trained via a multi-task objective: 

$$
\mathcal {L} _ {t o t a l} = \mathcal {L} _ {B C E} + \mathcal {L} _ {M S E} + \lambda_ {a w} \mathcal {L} _ {a w} + \lambda_ {e} \mathcal {L} _ {e} (9)
$$

where $\mathcal { L } _ { B C E }$ and $\mathcal { L } _ { M S E }$ target anomaly classification and sequence reconstruction, the detailed computation is provided in Appendix B.4. λ is the weights of different L. 

# 4. Experiments

# 4.1. Experimental Setup

Datasets and Metrics: Our evaluation is conducted on a comprehensive suite of 11 public time-series anomaly detection datasets collected from the TSB-AD benchmark (Liu & Paparrizos, 2024), covering a wide range of real-world and synthetic scenarios. We evaluate model performance using four standard metrics: Affiliation-F1, F1-T, Standard-F1, and VUS-PR. 

Baselines: We compare our method against three primary settings: (1) Zero-Shot TSFMs, which include TimeRCD (Lan et al., 2025), DADA (Shentu et al., 2024a), TS-Pulse (Ekambaram et al., 2025), MOMENT (Goswami et al., 2024), TimesFM (Das et al., 2024), Chronos (Ansari et al., 2024), and TimeMOE (Shi et al., 2024)). (2) Full-shot models, which include deep learning methods (TranAD (Tuli et al., 2022), USAD (Audibert et al., 2020), OmniAnomaly (Su et al., 2019)) and classical statistical algorithms (LOF (Breunig et al., 2000), IForest(Liu et al., 2008)). (3) Visionbased models, which includes VIT4TS and VLM4TS (He et al., 2025), VisualTimeAnomaly (Xu et al., 2025), and AnomLLM (Zhou & Yu, 2024)). The remainder of this section presents the main results, ablation study and model analysis. Results on multivariate datasets and additional details are in Appendix B. 

Comparison with time-series models: To evaluate the effectiveness of VETime, we conducted extensive experiments on 11 public univariate datasets, benchmarking against zeroshot and full-shot baselines. Crucially, VETime operates strictly in a zero-shot manner across all settings. As shown in Table 1, VETime demonstrates superior generalization, securing 25 out of 44 first-place rankings in the zero-shot setting and remarkably maintaining dominance with 23 firstplace rankings against full-shot methods. Moreover, VE-Time consistently achieves the lowest average ranks of 2.05 (zero-shot) and 2.02 (full-shot). It confirms that VETime provides not only higher detection accuracy but also stable performance advantages across diverse data domains compared to existing methods. The multivariate comparison results are provided in Appendix C.1. 

Comparison with vision-based models: Due to the high computational cost and token consumption of vision-based models, we evaluate VETime against them on only four public datasets, following the protocol of VLM4TS(He et al., 2025). As presented in Table 2, VETime demonstrates a decisive advantage in both detection accuracy and computational efficiency. VETime consistently achieves superior performance, notably far surpassing competing methods on the YAHOO dataset. Furthermore, VETime proves to be orders of magnitude more efficient, approximately 100 times faster than vision-based counterparts, confirming VE-Time as a highly practical solution for real-time anomaly detection tasks. 


Table 1. Performance of VETime against zero-shot and full-shot baselines in 11 public univariate datasets. The models marked with (†) were excluded where necessary due to potential data leakage under zero-shot setting. Asterisked (*) results are excluded from ranking due to data leaking. Red: best,Blue: second best.


<table><tr><td rowspan="2">Metric</td><td rowspan="2">Model</td><td colspan="11">Univariate Datasets</td><td rowspan="2">Total 1st</td><td rowspan="2">Total 2nd</td><td rowspan="2">Avg Rank</td></tr><tr><td>IOPS</td><td>MGAB</td><td>NAB</td><td>NEK</td><td>Power</td><td>SED</td><td>Stock</td><td>TODS</td><td>UCR</td><td>WSD</td><td>YAHOO</td></tr><tr><td colspan="16">Zero-Shot Models</td></tr><tr><td rowspan="8">Affiliation-F1</td><td>VETime</td><td>90.53</td><td>68.03</td><td>88.57</td><td>79.56</td><td>78.78</td><td>97.31</td><td>69.60</td><td>85.85</td><td>85.06</td><td>94.31</td><td>97.15</td><td>04</td><td>04</td><td>2.55</td></tr><tr><td>TimeRCD</td><td>83.28</td><td>70.69</td><td>82.48</td><td>79.73</td><td>85.51</td><td>96.87</td><td>71.84</td><td>86.37</td><td>84.63</td><td>90.33</td><td>96.65</td><td>02</td><td>03</td><td>3.27</td></tr><tr><td>DADA†</td><td>89.37*</td><td>67.66*</td><td>86.56</td><td>95.40</td><td>69.79</td><td>65.18</td><td>98.77</td><td>76.89</td><td>72.21</td><td>93.92</td><td>92.20*</td><td>02</td><td>00</td><td>4.00</td></tr><tr><td>TS-Pulse</td><td>68.76</td><td>67.33</td><td>70.80</td><td>73.05</td><td>69.94</td><td>67.44</td><td>67.93</td><td>67.90</td><td>67.70</td><td>68.22</td><td>70.05</td><td>00</td><td>00</td><td>6.64</td></tr><tr><td>MOMENT†</td><td>87.54*</td><td>66.76*</td><td>90.45*</td><td>92.26</td><td>75.97</td><td>59.13</td><td>45.26</td><td>59.76</td><td>75.77</td><td>95.39</td><td>79.99*</td><td>02</td><td>00</td><td>4.73</td></tr><tr><td>TimesFM</td><td>81.88</td><td>66.95</td><td>79.73</td><td>90.49</td><td>69.88</td><td>67.14</td><td>97.53</td><td>89.08</td><td>70.03</td><td>78.97</td><td>91.28</td><td>00</td><td>02</td><td>5.27</td></tr><tr><td>Chronos</td><td>90.12</td><td>67.89</td><td>86.66</td><td>93.63</td><td>69.72</td><td>67.89</td><td>96.85</td><td>91.96</td><td>74.35</td><td>90.98</td><td>96.34</td><td>01</td><td>02</td><td>3.27</td></tr><tr><td>Time MOE</td><td>76.34</td><td>67.23</td><td>80.51</td><td>80.50</td><td>71.19</td><td>60.98</td><td>63.28</td><td>54.68</td><td>73.56</td><td>80.25</td><td>69.70</td><td>00</td><td>00</td><td>6.27</td></tr><tr><td rowspan="8">F1.T</td><td>VETime</td><td>46.15</td><td>2.00</td><td>44.31</td><td>60.86</td><td>20.16</td><td>74.35</td><td>22.41</td><td>68.33</td><td>36.25</td><td>50.42</td><td>91.54</td><td>07</td><td>03</td><td>1.91</td></tr><tr><td>TimeRCD</td><td>28.44</td><td>01.81</td><td>38.85</td><td>35.87</td><td>28.47</td><td>69.43</td><td>31.73</td><td>65.89</td><td>34.30</td><td>35.04</td><td>85.86</td><td>01</td><td>06</td><td>3.27</td></tr><tr><td>DADA†</td><td>42.50*</td><td>00.91*</td><td>37.24</td><td>47.98</td><td>19.80</td><td>09.56</td><td>95.49</td><td>35.18</td><td>07.22</td><td>48.46</td><td>79.52*</td><td>01</td><td>05</td><td>4.45</td></tr><tr><td>TS-Pulse</td><td>04.10</td><td>00.81</td><td>34.61</td><td>27.07</td><td>19.90</td><td>09.71</td><td>15.98</td><td>13.45</td><td>05.12</td><td>4.57</td><td>05.50</td><td>01</td><td>01</td><td>7.00</td></tr><tr><td>MOMENT†</td><td>33.15*</td><td>00.80*</td><td>52.27*</td><td>63.66</td><td>19.91</td><td>09.54</td><td>18.04</td><td>17.47</td><td>13.02</td><td>41.98</td><td>11.69*</td><td>01</td><td>00</td><td>5.00</td></tr><tr><td>TimesFM</td><td>48.95</td><td>00.93</td><td>36.74</td><td>36.63</td><td>19.80</td><td>09.58</td><td>88.94</td><td>51.13</td><td>41.38</td><td>10.78</td><td>83.46</td><td>01</td><td>01</td><td>4.18</td></tr><tr><td>Chronos</td><td>45.45</td><td>01.10</td><td>36.10</td><td>33.16</td><td>19.90</td><td>13.18</td><td>89.3</td><td>53.90</td><td>10.88</td><td>39.82</td><td>79.00</td><td>00</td><td>00</td><td>4.27</td></tr><tr><td>Time MOE</td><td>25.95</td><td>00.63</td><td>38.70</td><td>15.78</td><td>19.85</td><td>17.73</td><td>34.13</td><td>20.91</td><td>08.29</td><td>22.6</td><td>37.11</td><td>00</td><td>00</td><td>5.91</td></tr><tr><td rowspan="8">Standard-F1</td><td>VETime</td><td>34.56</td><td>1.89</td><td>37.87</td><td>53.95</td><td>20.12</td><td>75.02</td><td>23.83</td><td>69.13</td><td>30.61</td><td>46.55</td><td>88.93</td><td>08</td><td>02</td><td>1.73</td></tr><tr><td>TimeRCD</td><td>24.22</td><td>01.62</td><td>27.70</td><td>33.05</td><td>28.59</td><td>69.88</td><td>32.61</td><td>67.02</td><td>28.13</td><td>31.96</td><td>87.02</td><td>01</td><td>06</td><td>3.27</td></tr><tr><td>DADA†</td><td>32.76*</td><td>00.80*</td><td>26.91</td><td>48.24</td><td>15.99</td><td>02.69</td><td>95.59</td><td>28.18</td><td>03.36</td><td>45.06</td><td>79.30*</td><td>01</td><td>05</td><td>4.36</td></tr><tr><td>TS-Pulse</td><td>03.54</td><td>00.73</td><td>21.61</td><td>23.96</td><td>18.27</td><td>08.84</td><td>15.46</td><td>12.45</td><td>02.05</td><td>2.17</td><td>04.00</td><td>00</td><td>00</td><td>6.82</td></tr><tr><td>MOMENT†</td><td>30.69*</td><td>00.67*</td><td>44.75*</td><td>63.85</td><td>16.39</td><td>03.36</td><td>19.38</td><td>14.64</td><td>09.00</td><td>41.42</td><td>10.54*</td><td>01</td><td>00</td><td>4.73</td></tr><tr><td>TimesFM</td><td>34.28</td><td>00.83</td><td>26.46</td><td>38.15</td><td>16.73</td><td>02.96</td><td>89.13</td><td>40.08</td><td>07.86</td><td>38.5</td><td>84.44</td><td>01</td><td>01</td><td>4.27</td></tr><tr><td>Chronos</td><td>32.69</td><td>00.99</td><td>26.22</td><td>33.54</td><td>17.47</td><td>08.74</td><td>89.41</td><td>40.52</td><td>08.21</td><td>34.58</td><td>78.89</td><td>00</td><td>00</td><td>4.55</td></tr><tr><td>Time MOE</td><td>26.52</td><td>00.45</td><td>26.20</td><td>11.47</td><td>12.16</td><td>17.73</td><td>34.32</td><td>16.38</td><td>04.09</td><td>20.09</td><td>27.50</td><td>00</td><td>00</td><td>6.55</td></tr><tr><td rowspan="8">VUS-PR</td><td>VETime</td><td>30.79</td><td>0.70</td><td>32.98</td><td>57.54</td><td>11.78</td><td>89.05</td><td>76.08</td><td>94.20</td><td>24.99</td><td>33.85</td><td>88.61</td><td>06</td><td>04</td><td>2.00</td></tr><tr><td>TimeRCD</td><td>20.23</td><td>01.05</td><td>24.32</td><td>27.88</td><td>21.25</td><td>80.75</td><td>77.28</td><td>93.46</td><td>23.09</td><td>21.77</td><td>84.41</td><td>02</td><td>04</td><td>3.09</td></tr><tr><td>DADA†</td><td>24.97*</td><td>00.57*</td><td>24.73</td><td>46.85</td><td>10.61</td><td>06.42</td><td>99.51</td><td>64.83</td><td>02.94</td><td>33.42</td><td>70.74*</td><td>01</td><td>01</td><td>4.00</td></tr><tr><td>TS-Pulse</td><td>04.64</td><td>00.56</td><td>16.40</td><td>19.39</td><td>11.72</td><td>09.11</td><td>70.95</td><td>45.86</td><td>01.20</td><td>1.83</td><td>09.93</td><td>00</td><td>00</td><td>6.82</td></tr><tr><td>MOMENT†</td><td>37.35*</td><td>00.56*</td><td>45.38*</td><td>67.74</td><td>10.50</td><td>04.31</td><td>76.97</td><td>56.45</td><td>06.17</td><td>55.26</td><td>30.81*</td><td>02</td><td>00</td><td>3.73</td></tr><tr><td>TimesFM</td><td>19.56</td><td>00.58</td><td>24.01</td><td>35.02</td><td>10.44</td><td>06.13</td><td>98.39</td><td>72.89</td><td>06.03</td><td>21.57</td><td>86.78</td><td>00</td><td>02</td><td>4.45</td></tr><tr><td>Chronos</td><td>19.00</td><td>00.60</td><td>23.76</td><td>31.80</td><td>10.95</td><td>08.65</td><td>97.49</td><td>70.66</td><td>06.56</td><td>18.81</td><td>83.54</td><td>00</td><td>00</td><td>4.91</td></tr><tr><td>Time MOE</td><td>16.63</td><td>00.52</td><td>22.62</td><td>19.76</td><td>09.34</td><td>10.87</td><td>74.78</td><td>48.78</td><td>02.10</td><td>10.93</td><td>20.90</td><td>00</td><td>00</td><td>6.82</td></tr><tr><td colspan="13">Grand Total (Zero-Shot)</td><td>25</td><td>13</td><td>2.05</td></tr><tr><td colspan="16">Full-Shot Models</td></tr><tr><td rowspan="6">Affiliation-F1</td><td>VETime</td><td>90.53</td><td>68.03</td><td>88.57</td><td>79.56</td><td>78.78</td><td>97.31</td><td>69.60</td><td>85.85</td><td>85.06</td><td>94.31</td><td>97.15</td><td>07</td><td>00</td><td>1.91</td></tr><tr><td>TranAD</td><td>83.19</td><td>67.28</td><td>90.28</td><td>85.02</td><td>71.56</td><td>61.03</td><td>57.94</td><td>52.76</td><td>73.31</td><td>84.34</td><td>76.08</td><td>00</td><td>04</td><td>3.45</td></tr><tr><td>USAD</td><td>71.08</td><td>67.81</td><td>91.54</td><td>71.13</td><td>76.48</td><td>55.60</td><td>35.92</td><td>47.90</td><td>76.00</td><td>65.10</td><td>53.05</td><td>00</td><td>02</td><td>4.36</td></tr><tr><td>OmniAnomaly</td><td>80.32</td><td>67.35</td><td>92.35</td><td>86.30</td><td>78.16</td><td>61.26</td><td>75.24</td><td>50.73</td><td>73.53</td><td>78.02</td><td>71.31</td><td>03</td><td>01</td><td>3.00</td></tr><tr><td>LOF</td><td>81.06</td><td>68.44</td><td>75.75</td><td>84.74</td><td>66.76</td><td>63.85</td><td>69.74</td><td>60.58</td><td>73.53</td><td>81.29</td><td>75.63</td><td>00</td><td>04</td><td>3.09</td></tr><tr><td>IForest</td><td>52.81</td><td>68.82</td><td>39.84</td><td>71.15</td><td>00.00</td><td>70.09</td><td>0.06</td><td>44.17</td><td>50.56</td><td>41.24</td><td>33.30</td><td>01</td><td>00</td><td>5.09</td></tr><tr><td rowspan="6">F1.T</td><td>VETime</td><td>46.15</td><td>2.00</td><td>44.31</td><td>60.86</td><td>20.16</td><td>74.35</td><td>22.41</td><td>68.33</td><td>36.25</td><td>50.42</td><td>91.54</td><td>05</td><td>03</td><td>2.09</td></tr><tr><td>TranAD</td><td>22.63</td><td>01.65</td><td>37.28</td><td>69.97</td><td>22.36</td><td>09.57</td><td>16.73</td><td>13.51</td><td>07.75</td><td>20.94</td><td>08.41</td><td>00</td><td>00</td><td>4.18</td></tr><tr><td>USAD</td><td>20.99</td><td>04.07</td><td>61.46</td><td>70.64</td><td>28.23</td><td>09.54</td><td>16.86</td><td>20.85</td><td>14.63</td><td>14.18</td><td>09.35</td><td>03</td><td>02</td><td>3.18</td></tr><tr><td>OmniAnomaly</td><td>51.17</td><td>01.61</td><td>40.09</td><td>82.20</td><td>23.48</td><td>09.68</td><td>36.22</td><td>14.33</td><td>08.47</td><td>34.79</td><td>24.16</td><td>02</td><td>04</td><td>3.09</td></tr><tr><td>LOF</td><td>27.97</td><td>01.15</td><td>35.76</td><td>63.57</td><td>19.80</td><td>09.60</td><td>66.14</td><td>31.63</td><td>08.31</td><td>24.38</td><td>55.93</td><td>01</td><td>02</td><td>3.45</td></tr><tr><td>IForest</td><td>07.64</td><td>00.84</td><td>21.44</td><td>65.56</td><td>00.00</td><td>09.54</td><td>1.10</td><td>11.06</td><td>06.36</td><td>4.28</td><td>04.90</td><td>00</td><td>00</td><td>5.00</td></tr><tr><td rowspan="6">Standard-F1</td><td>VETime</td><td>34.56</td><td>1.89</td><td>37.87</td><td>53.95</td><td>20.12</td><td>75.02</td><td>23.83</td><td>69.13</td><td>30.61</td><td>56.55</td><td>88.93</td><td>05</td><td>02</td><td>2.00</td></tr><tr><td>TranAD</td><td>34.85</td><td>01.46</td><td>27.33</td><td>60.36</td><td>22.36</td><td>02.63</td><td>16.23</td><td>11.94</td><td>04.40</td><td>20.23</td><td>05.70</td><td>00</td><td>01</td><td>3.91</td></tr><tr><td>USAD</td><td>30.66</td><td>03.89</td><td>56.15</td><td>62.91</td><td>28.24</td><td>03.41</td><td>17.99</td><td>23.87</td><td>10.74</td><td>13.20</td><td>07.21</td><td>03</td><td>02</td><td>2.73</td></tr><tr><td>OmniAnomaly</td><td>47.05</td><td>01.44</td><td>28.81</td><td>74.03</td><td>23.50</td><td>00.43</td><td>38.59</td><td>12.65</td><td>05.11</td><td>29.57</td><td>21.40</td><td>02</td><td>03</td><td>3.36</td></tr><tr><td>LOF</td><td>30.28</td><td>01.05</td><td>24.04</td><td>56.92</td><td>12.18</td><td>04.11</td><td>66.20</td><td>25.77</td><td>04.70</td><td>22.62</td><td>48.95</td><td>01</td><td>03</td><td>3.55</td></tr><tr><td>IForest</td><td>08.37</td><td>00.73</td><td>29.41</td><td>58.10</td><td>19.77</td><td>03.81</td><td>16.91</td><td>13.35</td><td>04.09</td><td>2.07</td><td>03.20</td><td>00</td><td>00</td><td>5.09</td></tr><tr><td rowspan="6">VUS-PR</td><td>VETime</td><td>30.79</td><td>0.70</td><td>32.98</td><td>57.54</td><td>11.78</td><td>89.05</td><td>76.08</td><td>94.20</td><td>24.99</td><td>33.85</td><td>88.61</td><td>06</td><td>02</td><td>2.09</td></tr><tr><td>TranAD</td><td>21.61</td><td>00.64</td><td>24.82</td><td>61.63</td><td>13.04</td><td>05.75</td><td>78.08</td><td>47.33</td><td>02.25</td><td>12.20</td><td>25.78</td><td>00</td><td>01</td><td>3.73</td></tr><tr><td>USAD</td><td>16.58</td><td>00.75</td><td>55.03</td><td>58.53</td><td>18.68</td><td>04.37</td><td>74.53</td><td>56.36</td><td>08.85</td><td>10.00</td><td>14.15</td><td>03</td><td>02</td><td>3.27</td></tr><tr><td>OmniAnomaly</td><td>25.35</td><td>00.64</td><td>27.17</td><td>74.51</td><td>14.32</td><td>06.20</td><td>91.29</td><td>45.55</td><td>02.40</td><td>16.37</td><td>29.26</td><td>02</td><td>03</td><td>2.64</td></tr><tr><td>LOF</td><td>19.43</td><td>00.57</td><td>21.18</td><td>58.52</td><td>09.31</td><td>06.81</td><td>83.07</td><td>49.14</td><td>02.39</td><td>12.85</td><td>41.37</td><td>00</td><td>02</td><td>4.18</td></tr><tr><td>IForest</td><td>08.59</td><td>00.62</td><td>23.57</td><td>56.50</td><td>11.56</td><td>07.71</td><td>70.99</td><td>46.62</td><td>02.88</td><td>2.06</td><td>10.47</td><td>00</td><td>01</td><td>5.00</td></tr></table>

# 4.2. Ablation Study

Table 3 evaluates the contribution of each VETime component. A detailed ablation study is provided in Appendix C.3. 

Reversible Image Conversion (RIC): Removing RIC (reverting to line plots) causes the most severe performance collapse, with VUS-PR plummeting from 33.85% to 18.71% on WSD and from 32.98% to 24.05% on NAB. This proves that the proposed image conversion lays a solid foundation for subsequent alignment and fusion. 

Patch-Level Temporal Alignment (PTA): Replacing PTA with direct mapping restricts fusion to coarse semantic interactions. The resulting misalignment and loss of fine-grained local correspondences lead to a noticeable decline in detection capability, evidenced by the VUS-PR dropping to 28.59% on the NAB dataset. 

Anomaly Window Contrastive Learning (AWCL): Although constructing effective contrastive pairs is challenging on densely anomalous datasets like YAHOO, it can help the model distinguish normal from anomalous patterns more sharply, further improving detection precision in most cases. 

Task-Adaptive Multi-Modal Fusion (TMF): Eliminating TMF (reverting to concatenation) causes consistent performance regression across all benchmarks. This validates that the task-aware gating mechanism dynamically regulates feature integration, ensuring robust performance by balancing the conflicting demands of reconstruction and classification. 


Table 2. Performance of VETime against vision-based baselines in 4 public univariate datasets.


<table><tr><td>Metric</td><td>Method</td><td>NAB</td><td>YAHOO</td><td>SMAP</td><td>MSL</td></tr><tr><td rowspan="5">Affiliation-F1</td><td>VETime</td><td>88.57</td><td>97.15</td><td>98.83</td><td>92.78</td></tr><tr><td>VIT4TS</td><td>61.12</td><td>60.66</td><td>91.55</td><td>65.80</td></tr><tr><td>VLM4TS</td><td>70.37</td><td>64.25</td><td>94.76</td><td>66.15</td></tr><tr><td>VisualTimeAnomaly</td><td>62.46</td><td>61.03</td><td>91.65</td><td>86.20</td></tr><tr><td>AnomLLM</td><td>58.53</td><td>36.43</td><td>42.52</td><td>46.08</td></tr><tr><td rowspan="5">F1_T</td><td>VETime</td><td>44.31</td><td>91.54</td><td>70.21</td><td>50.38</td></tr><tr><td>VIT4TS</td><td>28.29</td><td>7.61</td><td>50.97</td><td>23.48</td></tr><tr><td>VLM4TS</td><td>30.75</td><td>9.10</td><td>57.13</td><td>37.50</td></tr><tr><td>VisualTimeAnomaly</td><td>41.64</td><td>3.02</td><td>14.80</td><td>13.84</td></tr><tr><td>AnomLLM</td><td>17.76</td><td>1.67</td><td>4.17</td><td>3.32</td></tr><tr><td rowspan="5">Standard-F1</td><td>VETime</td><td>37.87</td><td>88.93</td><td>64.18</td><td>52.18</td></tr><tr><td>VIT4TS</td><td>33.81</td><td>7.74</td><td>51.65</td><td>27.05</td></tr><tr><td>VLM4TS</td><td>34.78</td><td>9.16</td><td>57.56</td><td>42.21</td></tr><tr><td>VisualTimeAnomaly</td><td>31.72</td><td>3.63</td><td>16.19</td><td>16.96</td></tr><tr><td>AnomLLM</td><td>23.70</td><td>2.36</td><td>6.65</td><td>10.70</td></tr><tr><td rowspan="5">VUS-PR</td><td>VETime</td><td>32.98</td><td>88.61</td><td>60.67</td><td>42.81</td></tr><tr><td>VIT4TS</td><td>25.78</td><td>31.63</td><td>46.44</td><td>26.52</td></tr><tr><td>VLM4TS</td><td>27.89</td><td>33.09</td><td>52.28</td><td>33.57</td></tr><tr><td>VisualTimeAnomaly</td><td>26.87</td><td>13.13</td><td>11.94</td><td>17.91</td></tr><tr><td>AnomLLM</td><td>19.50</td><td>10.44</td><td>4.92</td><td>9.77</td></tr><tr><td rowspan="5">Per-series runtime(s)</td><td>VETime</td><td>0.04</td><td>0.02</td><td>0.04</td><td>0.02</td></tr><tr><td>VIT4TS</td><td>1.72</td><td>5.86</td><td>2.23</td><td>2.75</td></tr><tr><td>VLM4TS</td><td>6.78</td><td>9.27</td><td>7.00</td><td>7.39</td></tr><tr><td>VisualTimeAnomaly</td><td>2.52</td><td>2.35</td><td>2.34</td><td>2.85</td></tr><tr><td>AnomLLM</td><td>3.59</td><td>3.91</td><td>3.40</td><td>3.95</td></tr></table>


Table 3. Performance comparison across models on NAB, WSD, and YAHOO datasets. RIC: Reversible Image Conversion, PTA: Patch-Level Temporal Alignment, AWCL: Anomaly Window Contrastive Learning, TMF: Task-Adaptive Multi-Modal Fusion.


<table><tr><td rowspan="2">Model</td><td colspan="2">NAB</td><td colspan="2">WSD</td><td colspan="2">YAHOO</td></tr><tr><td>Affiliation -F1</td><td>VUS -PR</td><td>Affiliation -F1</td><td>VUS -PR</td><td>Affiliation -F1</td><td>VUS -PR</td></tr><tr><td>w/o RIC</td><td>83.85</td><td>24.05</td><td>88.01</td><td>18.71</td><td>94.81</td><td>82.53</td></tr><tr><td>w/o PTA</td><td>85.66</td><td>28.59</td><td>91.62</td><td>30.57</td><td>96.97</td><td>86.54</td></tr><tr><td>w/o AWCL</td><td>86.84</td><td>31.23</td><td>91.63</td><td>29.00</td><td>97.11</td><td>87.26</td></tr><tr><td>w/o TMF</td><td>83.99</td><td>29.52</td><td>90.11</td><td>26.30</td><td>96.86</td><td>85.31</td></tr><tr><td>VETime</td><td>88.57</td><td>32.98</td><td>94.31</td><td>33.85</td><td>97.15</td><td>88.61</td></tr></table>

# 4.3. Model Analysis

Sensitivity Analysis of Hyperparameters: We evaluate the hyperparameters $\lambda _ { a w } , \lambda _ { e }$ , and τ on the NAB and WSD datasets, as shown in Figure $5 . \lambda _ { a w }$ and τ are key parameters in contrastive learning, and both excessively high and low values of these parameters result in performance degradation. Specifically, increasing $\lambda _ { a w }$ or τ beyond the optimal range tends to over-smooth the feature distribution. Additionally, λe is crucial for preventing expert collapse. When it is insufficient (< 0.2), the router struggles to balance the different modalities and tasks, weakening the model’s discriminative capability. Consequently, fixing these parameters at their optimal values yields robust performance across all datasets. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/fd86fce009541a080b64027a1139664c53a5114ca95d014be2a7137f84096bce.jpg)



Figure 5. Hyperparameter analysis of $\lambda _ { a w } , \lambda _ { e }$ , and τ in Terms of VUS-PR (%)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/4546a5d246c59f1df3223bd885b624d1b74eecfa9b016a1dce970911b4a25661.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/6dee930d67d2ffd4a295c178a19ee4055a8d5dda90fd7c62687bf1f8cda16d5d.jpg)



Figure 6. Visualization of fusion weights on the univariate datasets.


Visualization of Fusion Weights: To verify the Task-Adaptive Multi-Modal Fusion, we visualize weight distributions for detection and reconstruction heads (Figure 6). A distinct task-dependent divergence emerges: the router assigns higher weights to Anomaly-Enhanced features for detection to exploit their contrastive-learned discriminative power, while Temporal features dominate reconstruction to preserve numerical continuity. This dynamic recalibration confirms that the model effectively mitigates expert collapse by routing task-specific information streams—prioritizing high-level semantics for detection and low-level precision for reconstruction. 

Qualitative Results: Figure 7 visualizes anomaly scores across Point, Contextual, and Mixed scenarios to validate model adaptability. VETime consistently outperforms unimodal baselines: it maintains high-frequency details for precise point anomaly localization, avoiding ViT4TS’s oversmoothing while leveraging global visual semantics to identify long-duration contextual deviations. In complex mixed scenarios, VETime significantly suppresses background noise compared to TimeRCD and rectifies ViT4TS’s missed detections, demonstrating robust generalization across diverse anomaly patterns. 

# 5. Conclusion

We propose VETime, which synergizes temporal sensitivity and global visual context. To bridge the gap between heterogeneous modalities, we introduce a Reversible Image Conversion method combined with Patch-Level Temporal Alignment. Furthermore, by leveraging Anomaly Window Contrastive Learning and Task-Adaptive Multi-Modal Fusion, the model addresses the limitations of unimodal approaches in capturing diverse anomaly patterns. Extensive experiments verify that VETime consistently outperforms state-of-the-art baselines in zero-shot scenarios. Future work will incorporate textual modalities to enhance interpretability, evolving the framework to detect anomalies and provide semantic explanations for their underlying causes. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/688acbcca76643cf0a261385c088b258a439a6bca29ff2d194ab7a9100484e94.jpg)



Figure 7. Qualitative comparison of anomaly detection results. The top row displays the original time series data as blue lines and highlights ground-truth anomalies using red shaded regions. The subsequent rows illustrate the anomaly scores generated by TimeRCD, VLM4TS, and VETime.


# References



Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., et al. Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815, 2024. 





Audibert, J., Michiardi, P., Guyard, F., Marti, S., and Zuluaga, M. A. Usad: Unsupervised anomaly detection on multivariate time series. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 3395–3404, 2020. 





Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J. Lof: identifying density-based local outliers. In Proceedings of the 2000 ACM SIGMOD international conference on Management of data, pp. 93–104, 2000. 





Chen, M., Shen, L., Li, Z., Wang, X. J., Sun, J., and Liu, C. VisionTS: Visual masked autoencoders are free-lunch zero-shot time series forecasters. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 8979–9007, 2025. 





Das, A., Kong, W., Sen, R., and Zhou, Y. A decoderonly foundation model for time-series forecasting. In Forty-first International Conference on Machine Learning, 2024. 





Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, 





N. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), 2021. 





Ekambaram, V., Kumar, S., Jati, A., Mukherjee, S., Sakai, T., Dayama, P., Gifford, W. M., and Kalagnanam, J. Tspulse: Dual space tiny pre-trained models for rapid time-series analysis. arXiv preprint arXiv:2505.13033, 2025. 





Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., and Dubrawski, A. Moment: A family of open timeseries foundation models. In International Conference on Machine Learning (ICML), 2024. 





He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, R.´ Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16000–16009, 2022. 





He, Z., Alnegheimish, S., and Reimherr, M. Harnessing vision-language models for time series anomaly detection. arXiv preprint arXiv:2506.06836, 2025. 





Huet, A., Navarro, J. M., and Rossi, D. Local evaluation of time series anomaly detection algorithms. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 635–645, 2022. 





Hundman, K., Constantinou, V., Laporte, C., Colwell, I., and Soderstrom, T. Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 387–395, 2018. 





Lan, T., Le, H. D., Li, J., He, W., Wang, M., Liu, C., and Zhang, C. Towards foundation models for zero-shot time series anomaly detection: Leveraging synthetic data and relative context discrepancy, 2025. URL https:// arxiv.org/abs/2509.21190. 





Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann. 





Liu, D. and et al. Teaching time series to see and speak: Forecasting with aligned visual and textual perspectives. arXiv preprint arXiv:2506.24124, 2025a. 





Liu, F. T., Ting, K. M., and Zhou, Z.-H. Isolation forest. In 2008 eighth ieee international conference on data mining, pp. 413–422. IEEE, 2008. 





Liu, Q. and et al. Mllm4ts: Leveraging vision and multimodal language models for general time-series analysis. arXiv preprint arXiv:2510.07513, 2025b. 





Liu, Q. and Paparrizos, J. The elephant in the room: Towards a reliable time-series anomaly detection benchmark. In NeurIPS 2024, 2024. 





Ni, J., Zhao, Z., Shen, C., Tong, H., Song, D., Cheng, W., Luo, D., and Chen, H. Harnessing vision models for time series analysis: A survey. arXiv preprint arXiv:2502.08869, 2025. 





Paparrizos, J., Boniol, P., Palpanas, T., Tsay, R. S., Elmore, A., and Franklin, M. J. Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection. Proceedings of the VLDB Endowment, 15(11): 2774–2787, 2022. 





Ruan, Y. and Zhong, H. Vision-enhanced time series forecasting via latent diffusion models. arXiv preprint arXiv:2411.12345, 2024. 





Sarfraz, M. S., Chen, M.-Y., Layer, L., Peng, K., and Koulakis, M. Position: Quo vadis, unsupervised time series anomaly detection? In International Conference on Machine Learning, pp. 43461–43476. PMLR, 2024. 





Shentu, Q., Li, B., Zhao, K., Shu, Y., Rao, Z., Pan, L., Yang, B., and Guo, C. Towards a general time series anomaly detector with adaptive bottlenecks and dual adversarial decoders. arXiv preprint arXiv:2405.15273, 2024a. 





Shentu, Y., Liu, Z., Xie, Z., and Yu, Y. Dada: A general transformer-based time-series anomaly detection model. arXiv preprint arXiv:2404.04104, 2024b. 





Shi, X., Wang, S., Nie, Y., Li, D., Ye, Z., Wen, Q., and Jin, M. Time-moe: Billion-scale time series foundation models with mixture of experts. arXiv preprint arXiv:2409.16040, 2024. 





Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., and Pei, D. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In Proceedings of the 25th ACM SIGKDD international conference 





on knowledge discovery & data mining, pp. 2828–2837, 2019. 





Tuli, S., Casale, G., and Jennings, N. R. Tranad: Deep transformer networks for anomaly detection in multivariate time series data. Proceedings of the VLDB Endowment, 15(6):1201–1214, 2022. 





Wang, S., Li, J., Shi, X., Ye, Z., Mo, B., Lin, W., Ju, S., Chu, Z., and Jin, M. Timemixer++: A general time series pattern machine for universal predictive analysis. arXiv preprint arXiv:2410.16032, 2024. 





Woo, G., Liu, C., Kumar, A., Xiong, C., Savarese, S., and Sahoo, D. Unified training of universal time series forecasting transformers. In Forty-first International Conference on Machine Learning, 2024. 





Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., and Long, M. Timesnet: Temporal 2d-variation modeling for general time series analysis. In International Conference on Learning Representations, 2023. 





Xu, X., Wang, H., Liang, Y., Yu, P. S., Zhao, Y., and Shu, K. Can multimodal LLMs perform time series anomaly detection? arXiv preprint arXiv:2502.17812, 2025. 





Yang, H. and et al. Vitime: A visual intelligence-based foundation model for time series forecasting. arXiv preprint arXiv:2410.05264, 2024. 





Zeng, A., Chen, M., Zhang, L., and Xu, Q. Are transformers effective for time series forecasting? 2023. 





Zhong, S., Ruan, W., Jin, M., Li, H., Wen, Q., and Liang, Y. Time-vlm: Exploring multimodal vision-language models for augmented time series forecasting. In Proceedings of the 42nd International Conference on Machine Learning, 2025. 





Zhou, Z. and Yu, R. Can LLMs understand time series anomalies? arXiv preprint arXiv:2410.05440, 2024. 





Zhuang, Y., Liu, H., Liang, Y., Ma, C., Zhang, H., Nijssen, S., and Wang, W. See it, think it, sorted: Large multimodal models are few-shot time series anomaly analyzers. arXiv preprint arXiv:2411.02465, 2024. 



# A. Multivariate Model Adaptations

In the multivariate setting, we adapt the model as follows: 

For multivariate time series input, we extend the aforementioned univariate Reversible Image Conversion to construct a unified three-channel heatmap representation. Specifically, each variable is first decomposed via Seasonal-Trend decomposition, and subsequently transformed through a variable-specific Gamma correction coefficient to apply differentiated non-linear mappings. This strategy enhances inter-variable discriminability in the visual domain by adapting to their distinct statistical distributions. The resulting individual heatmaps are then concatenated along the temporal dimension to form an integrated multivariate heatmap of size $1 \times ( N _ { v } L ) \times 3$ , where $N _ { v }$ denotes the number of variables and L the temporal length per variable. Finally, this composite representation undergoes the same folding and scaling operations as described in the main text, yielding a consistent and unified visual input. 

Regarding Patch-Level Temporal Alignment, the folding and scaling operations applied to the image features remain unchanged in the multivariate setting, the output is $\hat { F } _ { V } \in \mathbb R ^ { N _ { T S } \times N _ { v } \times D _ { V } ^ { * } }$ . However, we inject an identical positional encoding vector $E _ { P O S }$ into the token sequence of each variable. Furthermore, we adapt the self-attention mechanism by leveraging the any-variate attention (Woo et al., 2024), which processes each variable’s temporal dynamics correction independently. Formally, the output are formulated as $F _ { V } \in \mathbb { R } ^ { ( N _ { T S } ^ { \bullet } N _ { v } ) \times D _ { T S } }$ . This design enables the model to separately reorganize the temporal order of features for each variable. 

The Anomaly Window Contrastive Learning objective is applied independently to the feature sequence of each variable. Specifically, for every variable, we construct its own anomaly context window and perform positive/negative sample selection within its own temporal trajectory. 

Finally, the Task-Adaptive Multi-Modal Fusion module remains unmodified in the multivariate setting, preserving its original architecture and functionality. 

# B. Experimental Details

# B.1. Benchmark Datasets

Our evaluation uses a selection of datasets from one primary source, the TSB-AD benchmark by (Liu & Paparrizos, 2024). 

• Univariate Datasets: We utilize a diverse collection of univariate datasets including IOPS, MGAB, NAB, NEK, Power, SED, TODS, UCR, and YAHOO. 

• Multivariate Datasets: For multivariate anomaly detection, we use the SWaT, SMAP, SMD, MSL, and PSM datasets. These are sourced from industrial control systems and spacecraft telemetry, presenting complex, multi-dimensional dependencies. 

The specific characteristics of these datasets, including their domain, number of time series (TS), average length, and anomaly ratio, are summarized in Table 4 and Table 5. 


Table 4. Univariate Datasets


<table><tr><td>Name</td><td>Domain</td><td>#TS</td><td>Avg Length</td><td>AR (%)</td></tr><tr><td>UCR</td><td>Misc.</td><td>228</td><td>67818.7</td><td>0.6</td></tr><tr><td>NAB</td><td>Web</td><td>28</td><td>5099.7</td><td>10.6</td></tr><tr><td>YAHOO</td><td>Web</td><td>259</td><td>1560.2</td><td>0.6</td></tr><tr><td>IOPS</td><td>Operations</td><td>17</td><td>72792.3</td><td>1.3</td></tr><tr><td>MGAB</td><td>Sensor</td><td>9</td><td>97777.8</td><td>0.2</td></tr><tr><td>SED</td><td>Energy</td><td>3</td><td>23332.3</td><td>4.1</td></tr><tr><td>TODS</td><td>Traffic</td><td>15</td><td>5000.0</td><td>6.3</td></tr><tr><td>NEK</td><td>Weather</td><td>9</td><td>1073.0</td><td>8.0</td></tr><tr><td>Power</td><td>Power Grid</td><td>1</td><td>35040.0</td><td>8.5</td></tr><tr><td>Stock</td><td>Stock</td><td>20</td><td>15000.0</td><td>9.4</td></tr><tr><td>WSD</td><td>Web</td><td>111</td><td>17444.5</td><td>0.6</td></tr></table>


Table 5. Multivariate Datasets


<table><tr><td>Name</td><td>Domain</td><td>#TS</td><td>Avg Length</td><td>AR (%)</td></tr><tr><td>MSL</td><td>Space</td><td>16</td><td>3119.4</td><td>5.1</td></tr><tr><td>PSM</td><td>Sensor</td><td>1</td><td>217624.0</td><td>11.2</td></tr><tr><td>SMAP</td><td>Space</td><td>27</td><td>7855.9</td><td>2.9</td></tr><tr><td>SMD</td><td>Server</td><td>22</td><td>25466.4</td><td>3.8</td></tr><tr><td>SWaT</td><td>ICS</td><td>2</td><td>207457.5</td><td>12.7</td></tr></table>

# B.2. Baselines

Zero-Shot Models: These models are pre-trained on large-scale datasets and can be applied directly to new time series without fine-tuning. 

• Time-RCD: This model, from the paper by (Lan et al., 2025), is a pre-trained general anomaly detector. We set the window size to 5000 and patch size is 16. 

• DADA: This model, from the paper by (Shentu et al., 2024a), is a pre-trained general anomaly detector. We set the window size to 100. 

• TS-Pulse: An IBM foundation model that uses a TSMixer architecture (Ekambaram et al., 2025). It is pre-trained on a massive corpus for multi-task applications, including anomaly detection. Following the suggestion from the paper, we set the window size with a length of 96. 

• MOMENT: A foundation model that utilizes a patch-based pre-training strategy to learn rich representations from diverse time series data (Goswami et al., 2024). We use a window size of 64. 

• TimesFM: A decoder-only transformer model from Google trained on a large time series corpus using a patching strategy, enabling strong zero-shot generalization (Das et al., 2024). We set the window size to 96. 

• Chronos: A generative model that frames time series analysis as a language modeling task, using a transformer-based architecture to learn and predict time series values (Ansari et al., 2024). Its window size is 100. 

• Time MOE: A decoder-only transformer model with a sparse Mixture-of-Experts (MoE) architecture. It is pre-trained on a large time series corpus for forecasting and multi-task learning (Shi et al., 2024). We set the window size to 96. 

Full-Shot Models: These models require training on the target dataset. 

• TranAD: A transformer-based model that uses a reconstructive approach to detect anomalies by comparing original and reconstructed time series (Tuli et al., 2022). It is configured with a window size of 10. 

• USAD: An autoencoder-based model that employs an adversarial training mechanism to enhance its reconstruction capability and anomaly detection (Audibert et al., 2020). We set the window size to 100. 

• OmniAnomaly: A deep learning model that uses a Variational Autoencoder (VAE) with a Gated Recurrent Unit (GRU) to learn normal patterns and detect deviations (Su et al., 2019). The model is configured with a window size of 100. 

• LOF: A traditional statistical method that measures the local deviation of a data point from its neighbors, identifying anomalies with lower local density (Breunig et al., 2000). For univariate datasets, we set n neighbors to 50, and for multivariate, we set n neighbors to 50 with metric as euclidean. 

• IForest: An ensemble of Isolation Trees that isolates anomalies based on the number of random partitions required to separate them from the rest of the data (Liu et al., 2008). For univariate datasets, we set n estimators to 200, and for multivariate, we set n estimators to 25 and max features to 0.8. 

Vision-based Models These models use powerful pre trained models for temporal anomaly detection without the need for additional training. 

• VIT4TS: A lightweight anomaly screening module that uses a pretrained Vision Transformer to encode time series rendered as line plots from overlapping sliding windows (window length 224, stride 56) and computes cross-window patch-level dissimilarity against a median normal reference to generate high-resolution anomaly heatmaps without fine-tuning(He et al., 2025) . 

VLM4TS: A zero-shot time series anomaly detection framework that combines a pretrained vision-language model (GPT-4o) with a two-stage pipeline. It first using ViT4TS for candidate localization and then prompting the full VLM with a global line plot and textual candidate list to perform semantic validation and refinement of anomalies under a unified multimodal reasoning process(He et al., 2025). 

• VisualTimeAnomaly: It converts univariate and multivariate time series into line-plot images (TSIs) and uses off-theshelf multimodal large language models (GPT-4o) on point-wise, range-wise, and variate-wise anomaly detection tasks. (Xu et al., 2025). 

• AnomLLM: It involves plotting time series data as line charts and leverages GPT-4o-mini’s visual multimodal capabilities to detect anomaly patterns by directly ”viewing” the plot, rather than analyzing numerical text.(Zhou & Yu, 2024). 

# B.3. Evaluation Metric Calculations

Our performance evaluation is conducted using four metrics: Affiliation-F1, Temporal-F1 (F 1T ), Standard-F1 (F 1), and Volume Under Surface - Precision/Recall(VUS-PR). 

• Standard-F1 is a widely used metric that provides a harmonic mean of precision and recall. It is calculated using point-wise True Positives (TP), False Positives (FP), and False Negatives (FN). 

• F1-T as described by (Sarfraz et al., 2024), is a range-based metric that evaluates anomaly detection performance by considering the temporal context of anomalies. It is a variant of the F1-score that addresses common issues in time series evaluation, such as overlapping predictions and temporal proximity. 

• Affiliation-F1 is a distance-based metric that measures the ”affiliation” or proximity between the ground truth and predicted anomaly points. It is designed to be less sensitive to minor temporal shifts in the predicted anomalies. The score is calculated by finding the optimal one-to-one mapping between the ground truth and detected anomalies and then computing the F1-score based on these affiliations (Huet et al., 2022). 

• VUS-PR is a threshold-independent, parameter-free metric for time series anomaly detection. Unlike point-wise metrics, VUS-PR is robust to time lags and measures the area under a 3D surface plot of precision, recall, and a buffer parameter (Paparrizos et al., 2022). It addresses the limitations of standard F1 scores by creating a continuous buffer region around each anomaly, thus providing a more reliable and nuanced evaluation of model performance. 

# B.4. Implementation Details

Implementation: The input time series is divided into patches of size 16. The visual and temporal backbone employ the encoder of the frozen MAE(Chen et al., 2025) and pre-trained transformer architecture(Lan et al., 2025), respectively, with feature dimensions of 768 and 512. These are projected to a shared dimension of 512 during fusion. Hyper-parameters λaw, λe, and τ are fixed at 0.1, 0.2, and 0.1. The model is trained on synthetic data (Lan et al., 2025) with a batch size of 32, using AdamW optimizer with a learning rate of 5e-4 and a weight decay of 1e-5, for up to 25 epochs, with early stopping if no improvement is observed for 4 consecutive epochs. 

Encoders: Our framework leverages a dual-encoder architecture to process heterogeneous modalities. 

• Vision Encoder: We adopt the encoder from a pre-trained Masked Autoencoder (MAE) as our visual backbone. Specifically, we utilize the lightweight ViT-Base architecture to extract robust semantic representations from the converted time-series images. 

• Time-Series Encoder: We employ a classical Transformer architecture consisting of learnable positional encodings, a stack of Encoder layers with Multi-Head Self-Attention (MHSA), and Feed-Forward Networks (FFN), followed by Layer Normalization. This encoder features 8 transformer layers, each with 8 attention heads. We initialize our encoder 

using the pre-trained weights from Time-RCD (Lan et al., 2025). To achieve parameter-efficient adaptation, we apply Low-Rank Adaptation (LoRA) to fine-tune the linear projection matrices within both the Attention mechanism and the FFN modules. Specifically, we set the LoRA rank r = 8 and the scaling hyperparameter $\alpha = 1 6$ , resulting in a scaling factor of $\alpha / r = 2$ . LoRA adapters are inserted into all trainable linear layers of the encoder, while the original pre-trained weights remain frozen during fine-tuning. 

Loss: The task-specific features $F _ { A D }$ and $F _ { R e c } { \mathrm { : } }$ , derived from the fusion module, are passed through dedicated prediction heads to generate the final anomaly probabilities and reconstructed sequence: 

$$
\hat {y} _ {t} = \text { Softmax } (\mathbf {M L P} _ {A D} (F _ {A D})), \quad \hat {x} _ {t} = \mathbf {M L P} _ {R e c} (F _ {R e c}) \tag {10}
$$

where $\hat { Y } \in \mathbb { R } ^ { L }$ denotes the sequence of anomaly probabilities $\{ \hat { y } _ { t } \} _ { t = 1 } ^ { L }$ , and $\hat { X } \in \mathbb { R } ^ { L }$ represents the reconstructed time series $\{ \hat { x } _ { t } \} _ { t = 1 } ^ { L } . \mathrm { M L P } _ { A D }$ and ML $. \mathrm { P } _ { R e c }$ act as the anomaly classifier and reconstruction projector, respectively. $S o f t m a x ( \cdot )$ is the Softmax function used to map the logits to the range [0, 1] for probability estimation. 

The model is trained end-to-end using a composite loss function that balances classification accuracy, reconstruction quality, and feature alignment: 

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {B C E} + \mathcal {L} _ {M S E} + \lambda_ {a w} \mathcal {L} _ {a w} + \lambda_ {e} \mathcal {L} _ {e} \tag {11}
$$

where $\mathcal { L } _ { B C E }$ denotes the Binary Cross-Entropy loss for anomaly classification: 

$$
\mathcal {L} _ {B C E} = - \frac {1}{L} \sum_ {t = 1} ^ {L} \left[ y _ {t} \log (\hat {y} _ {t}) + (1 - y _ {t}) \log (1 - \hat {y} _ {t}) \right] \tag {12}
$$

where L is the sequence length, $y _ { t } \in \{ 0 , 1 \}$ is the ground truth anomaly label at time step t, and $\hat { y } _ { t }$ is the predicted anomaly probability. $\mathcal { L } _ { M S E }$ represents the Mean Squared Error loss supervising sequence reconstruction: 

$$
\mathcal {L} _ {M S E} = \frac {1}{L} \sum_ {t = 1} ^ {L} \| x _ {t} - \hat {x} _ {t} \| _ {2} ^ {2} \tag {13}
$$

where $x _ { t }$ and $\hat { x } _ { t }$ represent the original and reconstructed input values at time step $t ,$ respectively. 

Training Data: To ensure robust zero-shot generalization, we constructed a large-scale synthetic dataset comprising a total of 0.5 billion data points refer to (Lan et al., 2025). The dataset features sequences with variable lengths ranging from 500 to 10,000 time steps. It is designed to cover a diverse spectrum of anomaly patterns, encompassing both long-term context deviations (e.g., trend shifts) and fine-grained local anomalies (e.g., point outliers). 

# C. Additional Experimental Results

# C.1. Multivariate Experiments

As shown in Table 6, VETime establishes a new state-of-the-art in zero-shot anomaly detection, demonstrating overwhelming superiority over existing baselines (e.g., TimeRCD, DADA) by securing the top rank in 15 comparison instances. It consistently achieves the highest performance in critical metrics, particularly VUS-PR, across all datasets, which highlights its stability and precision independent of threshold selection. Remarkably, despite operating strictly in a zero-shot capacity, VETime exhibits exceptional generalization, frequently matching or even surpassing fully supervised full-shot models like TranAD and USAD (e.g., on MSL and SMAP). This cross-paradigm success validates the effectiveness of our proposed multi-modal alignment and fusion mechanism, proving its ability to capture universal discriminative anomaly patterns without the need for domain-specific training. 

# C.2. Ablation Study on Vision Encoder

Impact of vision backbones. To investigate the influence of the vision backbone on detection performance, we evaluate VETime with three vision encoders with different initialized pre-trained weights: ViT (Base) (Dosovitskiy et al., 2021), MAE (Base), and MAE (Large) (He et al., 2022). As shown in Table 7, MAE-based encoders consistently outperform the standard ViT (Base), demonstrating the benefit of masked autoencoding pre-training for time-series anomaly detection—a task that inherently involves reconstructing and reasoning about corrupted or anomalous segments. However, scaling up to MAE (Large) yields only marginal gains despite a 3.5× increase in parameters, and even degrades performance on YAHOO. In contrast, MAE (Base) delivers the best balance between performance, efficiency, and robustness across all three datasets. Therefore, we adopt MAE (Base) as the default vision encoder in our final VETime architecture. 


Table 6. Performance of VETime against zero-shot and full-shot baselines. VETime operates in a strictly zero-shot capacity in all comparisons. Best result is in red, second-best is in blue. Asterisked (*) results are excluded from ranking due to data leaking.


<table><tr><td rowspan="2">Metric</td><td rowspan="2">Model</td><td colspan="5">Multivariate Datasets</td><td rowspan="2">Total 1st</td><td rowspan="2">Total 2nd</td></tr><tr><td>MSL</td><td>PSM</td><td>SMAP</td><td>SMD</td><td>SWaT</td></tr><tr><td colspan="9">Zero-Shot Models</td></tr><tr><td rowspan="6">Affiliation-F1</td><td>VETime</td><td>87.02</td><td>77.27</td><td>87.06</td><td>88.94</td><td>75.45</td><td>01</td><td>03</td></tr><tr><td>TimeRCD</td><td>81.16</td><td>81.61</td><td>87.73</td><td>92.58</td><td>71.55</td><td>03</td><td>01</td></tr><tr><td>DADA†</td><td>76.57</td><td>81.27</td><td>76.92</td><td>83.74</td><td>76.18</td><td>01</td><td>01</td></tr><tr><td>TS-Pulse</td><td>70.14</td><td>70.28</td><td>69.21</td><td>68.21</td><td>71.18</td><td>00</td><td>00</td></tr><tr><td>MOMENT†</td><td>74.55*</td><td>65.79</td><td>77.42 *</td><td>74.00 *</td><td>70.17</td><td>00</td><td>00</td></tr><tr><td>Time MOE</td><td>69.85</td><td>54.74</td><td>74.38</td><td>69.97</td><td>64.37</td><td>00</td><td>00</td></tr><tr><td rowspan="6">F1_T</td><td>VETime</td><td>43.52</td><td>39.06</td><td>45.52</td><td>53.36</td><td>39.94</td><td>04</td><td>01</td></tr><tr><td>TimeRCD</td><td>42.47</td><td>37.98</td><td>33.74</td><td>53.91</td><td>30.28</td><td>01</td><td>03</td></tr><tr><td>DADA†</td><td>34.58</td><td>31.84</td><td>30.42</td><td>40.80</td><td>35.13</td><td>00</td><td>01</td></tr><tr><td>TS-Pulse</td><td>23.57</td><td>25.39</td><td>12.34</td><td>09.15</td><td>28.58</td><td>00</td><td>00</td></tr><tr><td>MOMENT†</td><td>25.97 *</td><td>27.77</td><td>17.93 *</td><td>28.68 *</td><td>28.76</td><td>00</td><td>00</td></tr><tr><td>Time MOE</td><td>23.92</td><td>26.82</td><td>14.22</td><td>19.90</td><td>30.11</td><td>00</td><td>00</td></tr><tr><td rowspan="6">Standard-F1</td><td>VETime</td><td>35.24</td><td>27.05</td><td>32.16</td><td>51.41</td><td>55.87</td><td>05</td><td>00</td></tr><tr><td>TimeRCD</td><td>30.66</td><td>26</td><td>30.48</td><td>44.89</td><td>28.73</td><td>00</td><td>04</td></tr><tr><td>DADA†</td><td>22.13</td><td>24.07</td><td>26.75</td><td>34.94</td><td>34.78</td><td>00</td><td>01</td></tr><tr><td>TS-Pulse</td><td>12.56</td><td>22.31</td><td>07.44</td><td>08.00</td><td>23.84</td><td>00</td><td>00</td></tr><tr><td>MOMENT†</td><td>14.43 *</td><td>23.83</td><td>12.92 *</td><td>29.78 *</td><td>21.30</td><td>00</td><td>00</td></tr><tr><td>Time MOE</td><td>12.85</td><td>24.80</td><td>09.01</td><td>21.62</td><td>23.58</td><td>00</td><td>00</td></tr><tr><td rowspan="6">VUS-PR</td><td>VETime</td><td>30.51</td><td>23.86</td><td>23.70</td><td>48.21</td><td>42.93</td><td>05</td><td>00</td></tr><tr><td>TimeRCD</td><td>20.45</td><td>18.69</td><td>22.68</td><td>37.03</td><td>17.58</td><td>00</td><td>04</td></tr><tr><td>DADA†</td><td>12.74</td><td>17.17</td><td>20.02</td><td>25.98</td><td>21.13</td><td>00</td><td>01</td></tr><tr><td>TS-Pulse</td><td>07.41</td><td>14.48</td><td>03.99</td><td>04.56</td><td>15.67</td><td>00</td><td>00</td></tr><tr><td>MOMENT†</td><td>09.32 *</td><td>16.48</td><td>08.97 *</td><td>15.96 *</td><td>14.90</td><td>00</td><td>00</td></tr><tr><td>Time MOE</td><td>07.82</td><td>15.68</td><td>04.98</td><td>11.12</td><td>16.20</td><td>00</td><td>00</td></tr><tr><td colspan="7">VETime Grand Total (Zero-Shot)</td><td>15</td><td>4</td></tr><tr><td colspan="9">Full-Shot Models</td></tr><tr><td rowspan="5">Affiliation-F1</td><td>VETime</td><td>87.02</td><td>77.27</td><td>87.06</td><td>88.94</td><td>75.45</td><td>03</td><td>02</td></tr><tr><td>TranAD</td><td>79.91</td><td>73.83</td><td>87.39</td><td>92.20</td><td>75.37</td><td>01</td><td>01</td></tr><tr><td>USAD</td><td>81.86</td><td>57.86</td><td>87.25</td><td>85.09</td><td>75.06</td><td>01</td><td>00</td></tr><tr><td>OmniAnomaly</td><td>83.15</td><td>58.17</td><td>91.38</td><td>85.82</td><td>73.39</td><td>03</td><td>02</td></tr><tr><td>LOF</td><td>84.35</td><td>61.98</td><td>63.32</td><td>64.13</td><td>56.34</td><td>00</td><td>01</td></tr><tr><td rowspan="5">F1_T</td><td>VETime</td><td>43.52</td><td>39.06</td><td>45.52</td><td>53.36</td><td>39.94</td><td>02</td><td>01</td></tr><tr><td>TranAD</td><td>39.42</td><td>25.49</td><td>29.12</td><td>37.98</td><td>49.58</td><td>00</td><td>01</td></tr><tr><td>USAD</td><td>48.71</td><td>28.96</td><td>43.94</td><td>50.41</td><td>50.41</td><td>01</td><td>01</td></tr><tr><td>OmniAnomaly</td><td>49.36</td><td>30.42</td><td>46.63</td><td>51.84</td><td>46.64</td><td>02</td><td>02</td></tr><tr><td>LOF</td><td>38.97</td><td>25.58</td><td>21.81</td><td>10.13</td><td>30.62</td><td>00</td><td>00</td></tr><tr><td rowspan="5">Standard-F1</td><td>VETime</td><td>35.24</td><td>27.05</td><td>32.16</td><td>51.41</td><td>55.87</td><td>00</td><td>00</td></tr><tr><td>TranAD</td><td>29.60</td><td>25.63</td><td>25.11</td><td>43.99</td><td>61.86</td><td>00</td><td>01</td></tr><tr><td>USAD</td><td>38.71</td><td>28.41</td><td>38.66</td><td>53.06</td><td>62.82</td><td>01</td><td>04</td></tr><tr><td>OmniAnomaly</td><td>39.10</td><td>30.43</td><td>40.50</td><td>57.06</td><td>55.93</td><td>04</td><td>00</td></tr><tr><td>LOF</td><td>30.65</td><td>18.80</td><td>18.70</td><td>08.41</td><td>29.08</td><td>00</td><td>00</td></tr><tr><td rowspan="5">VUS-PR</td><td>VETime</td><td>30.51</td><td>23.86</td><td>23.70</td><td>48.21</td><td>42.93</td><td>02</td><td>01</td></tr><tr><td>TranAD</td><td>14.78</td><td>16.49</td><td>13.37</td><td>28.34</td><td>47.37</td><td>01</td><td>00</td></tr><tr><td>USAD</td><td>29.95</td><td>17.59</td><td>26.37</td><td>34.53</td><td>44.73</td><td>00</td><td>02</td></tr><tr><td>OmniAnomaly</td><td>31.57</td><td>18.58</td><td>28.07</td><td>37.44</td><td>42.97</td><td>02</td><td>02</td></tr><tr><td>LOF</td><td>24.67</td><td>13.58</td><td>10.59</td><td>04.40</td><td>14.50</td><td>00</td><td>00</td></tr><tr><td colspan="7">VETime Grand Total (Full-Shot)</td><td>7</td><td>4</td></tr></table>

Impact of the different imaging strategies. The ablation study in Table 8 systematically evaluates the contribution of each component in our reversible image conversion pipeline. Strategy A, which uses a simple line plot without any advanced encoding, achieves relatively low VUS-PR scores, indicating limited capacity to capture subtle anomaly patterns. Strategy B introduces multi-channel intensity mapping, significantly improving both Affiliation-F1 and VUS-PR across all datasets by leveraging RGB channels to encode trend and residual components jointly. This demonstrates that richer visual representations enhance anomaly detection sensitivity. Strategy C adds adaptive folding, further boosting performance on WSD (Affiliation-F1: 90.55% → 92.30%), suggesting that spatial reorganization via periodic folding helps preserve long-term temporal dependencies. However, its impact is less pronounced on shorter sequences like Yahoo. Strategy D incorporates dimension-aware scaling, which improves generalization by aligning input resolution to standard vision models while preserving waveform fidelity. Notably, the full model (Strategy E) achieves the highest scores across all metrics and datasets, confirming the synergistic effect of combining multi-channel encoding, adaptive folding, and dimension-aware scaling. 


Table 7. Ablation study grouped by different vision encoder. Metrics: Affiliation-F1 (%) and VUS-PR (%).


<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="2">NAB</td><td colspan="2">WSD</td><td colspan="2">YAHOO</td></tr><tr><td>Affiliation-F1</td><td>VUS-PR</td><td>Affiliation-F1</td><td>VUS-PR</td><td>Affiliation-F1</td><td>VUS-PR</td></tr><tr><td>Vit(Base)</td><td>85.64M</td><td>88.42</td><td>32.72</td><td>93.01</td><td>30.90</td><td>95.87</td><td>86.12</td></tr><tr><td>MAE(Base)</td><td>85.64M</td><td>88.57</td><td>32.98</td><td>94.31</td><td>33.85</td><td>97.15</td><td>88.61</td></tr><tr><td>MAE(Large)</td><td>303.10M</td><td>88.67</td><td>31.54</td><td>93.94</td><td>34.10</td><td>97.03</td><td>87.20</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/e4f6bcd6efebfd8312cd315be13d087bacb6aa94fa065593f39643cf6c992b28.jpg)



(a) Example of a time series signal


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/ca336d0b35afaeb076fe553a99249646c59603f8911b78eac338ea9cc35214d5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/d5be1fde38ab3e8536eee662de86f55b87cff773966cabf36f66e1c4837d370e.jpg)



B


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/c87f99d442c18a988ce5bcd4c3d793abbe91388ef323a775db616e6c4afa33b8.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/8c4d8f7d78b23f4861468a6322c5c93e4cddfa09b5aba4035ca2bf84504c6d80.jpg)



D


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/5b02a7a2cdbd9c36d994c0d7e94dbc363dfb93ea13b9bb44571611113ccef162.jpg)



E



(b) Visualization of different imaging strategies.



Figure 8. Visualization of different imaging strategies.



Table 8. Ablation study grouped by different imaging strategies in Figure 8. Metrics: Affiliation-F1 (%) and VUS-PR (%).


<table><tr><td rowspan="2">Strategies</td><td rowspan="2">Description</td><td colspan="2">NAB</td><td colspan="2">WSD</td><td colspan="2">YAHOO</td></tr><tr><td>Affiliation-F1</td><td>VUS-PR</td><td>Affiliation-F1</td><td>VUS-PR</td><td>Affiliation-F1</td><td>VUS-PR</td></tr><tr><td>A</td><td>Line plot (w/o Reversible Image Conversion)</td><td>83.85</td><td>24.05</td><td>88.01</td><td>18.71</td><td>94.81</td><td>82.53</td></tr><tr><td>B</td><td>w/o Multi-Channel Intensity Mapping</td><td>86.47</td><td>28.81</td><td>91.25</td><td>33.61</td><td>96.53</td><td>87.30</td></tr><tr><td>C</td><td>w/o Adaptive Folding</td><td>85.07</td><td>25.03</td><td>90.55</td><td>25.03</td><td>94.71</td><td>84.11</td></tr><tr><td>D</td><td>w/o Dimension-Aware Scaling</td><td>86.11</td><td>30.33</td><td>92.30</td><td>30.79</td><td>95.57</td><td>86.45</td></tr><tr><td>E(Ours)</td><td>-</td><td>88.57</td><td>32.98</td><td>94.31</td><td>33.85</td><td>97.15</td><td>88.61</td></tr></table>

# C.3. Full Ablation Study on each component

To verify the effectiveness of each component in our proposed framework, we conducted a comprehensive ablation study on the NAB, WSD, and YAHOO datasets. Table 9 summarizes the results grouped by the three key modules: Patch-Level Temporal Alignment (PTA), Anomaly Window Contrastive Learning (AWCL), and Task-Adaptive Multi-Modal Fusion (TMF). 


Table 9. Ablation study grouped by model components: PTA (Patch-Level Temporal Alignment), AWCL (Anomaly Window Contrastive Learning), and TMF (Task-Adaptive Multi-Modal Fusion). Metrics: Affiliation-F1 (%) and VUS-PR (%).


<table><tr><td rowspan="2">Model</td><td rowspan="2">Component</td><td colspan="2">NAB</td><td colspan="2">WSD</td><td colspan="2">YAHOO</td></tr><tr><td>Affiliation-F1</td><td>VUS-PR</td><td>Affiliation-F1</td><td>VUS-PR</td><td>Affiliation-F1</td><td>VUS-PR</td></tr><tr><td rowspan="3">PTA</td><td>w/o adaptive folding</td><td>87.81</td><td>29.99</td><td>91.49</td><td>30.41</td><td>97.10</td><td>86.70</td></tr><tr><td>w/o positional encoding</td><td>88.16</td><td>30.27</td><td>92.29</td><td>31.18</td><td>97.25</td><td>86.81</td></tr><tr><td>w/o attention</td><td>85.69</td><td>29.88</td><td>91.57</td><td>29.46</td><td>96.92</td><td>85.53</td></tr><tr><td rowspan="2">AWCL</td><td>w/o inter-window contrast</td><td>86.69</td><td>31.17</td><td>92.85</td><td>30.54</td><td>97.15</td><td>87.29</td></tr><tr><td>w/o intra-window contrast</td><td>86.15</td><td>30.86</td><td>92.93</td><td>30.57</td><td>97.77</td><td>88.29</td></tr><tr><td rowspan="3">TMF</td><td>Concatenation</td><td>83.99</td><td>28.54</td><td>89.80</td><td>26.30</td><td>95.69</td><td>85.31</td></tr><tr><td>Addition</td><td>85.07</td><td>29.52</td><td>90.11</td><td>26.39</td><td>96.86</td><td>85.72</td></tr><tr><td>w/o reconstruction head</td><td>86.96</td><td>28.44</td><td>90.87</td><td>28.63</td><td>96.88</td><td>86.58</td></tr></table>

Effect of Patch-Level Temporal Alignment (PTA). We investigated the contributions of adaptive folding, positional encoding, and the attention mechanism within the PTA module. As observed, removing the attention mechanism resulted in the most significant performance degradation across all datasets (e.g., Affiliation-F1 dropped to 85.69% on NAB). This indicates that the attention mechanism is pivotal for capturing long-range dependencies and accurately aligning visual patches with temporal contexts. Furthermore, the exclusion of adaptive folding also led to a notable decline, validating its role in preserving intrinsic temporal structures during feature transformation. These results confirm that the complete PTA design is essential for ensuring fine-grained semantic correspondence. 

Impact of Anomaly Window Contrastive Learning (AWCL). The ablation results for AWCL demonstrate the necessity of combined contrastive strategies. The removal of intra-window contrast generally caused a more severe drop in detection accuracy compared to removing inter-window contrast, particularly on the NAB and WSD datasets. This suggests that learning discriminative representations within local windows is fundamental for identifying fine-grained anomalies. However, the absence of inter-window contrast also negatively impacted performance, highlighting its importance in distinguishing anomalous patterns from normal ones globally. The synergy of both strategies is crucial for maximizing the discriminability of anomaly information. 

Effectiveness of Task-Adaptive Multi-Modal Fusion (TMF). In the TMF module, we compared our proposed adaptive fusion against naive strategies (Concatenation and Addition) and examined the role of the reconstruction head. The results reveal that simple Concatenation and Addition yielded the poorest performance (e.g., Concatenation achieved only 83.99% Affiliation-F1 on NAB), proving that simple linear combinations fail to capture complex cross-modal interactions. Additionally, removing the reconstruction head led to a consistent performance drop. This empirical evidence supports the conclusion that sequence reconstruction serves as a critical auxiliary constraint, promoting deep feature interaction and enhancing the robustness of fused representations for the primary anomaly classification task. 


Table 10. Performance comparison under different types of anomalies.


<table><tr><td>Metric</td><td>Model</td><td>Local</td><td>Global</td><td>Mixed</td></tr><tr><td rowspan="3">Affiliation-F1</td><td>VETime</td><td>93.20</td><td>91.79</td><td>87.78</td></tr><tr><td>TimeRCD</td><td>74.84</td><td>73.84</td><td>73.03</td></tr><tr><td>VLM4TS</td><td>67.11</td><td>61.94</td><td>60.79</td></tr><tr><td rowspan="3">F1_T</td><td>VETime</td><td>66.17</td><td>63.32</td><td>52.61</td></tr><tr><td>TimeRCD</td><td>31.51</td><td>54.06</td><td>42.12</td></tr><tr><td>VLM4TS</td><td>21.60</td><td>35.90</td><td>29.63</td></tr><tr><td rowspan="3">Standard-F1</td><td>VETime</td><td>62.33</td><td>59.74</td><td>44.28</td></tr><tr><td>TimeRCD</td><td>28.34</td><td>32.18</td><td>26.20</td></tr><tr><td>VLM4TS</td><td>23.15</td><td>35.90</td><td>34.52</td></tr><tr><td rowspan="3">VUS-PR</td><td>VETime</td><td>59.33</td><td>57.59</td><td>44.61</td></tr><tr><td>TimeRCD</td><td>61.85</td><td>44.93</td><td>42.90</td></tr><tr><td>VLM4TS</td><td>30.57</td><td>32.10</td><td>25.61</td></tr></table>

![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/74401b651c69805dfb0e2b7bcb4a1fb95d3f55b94cddb9b3ca60c507ac28fc75.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/838caf60c9a7e2731942eff3e1794e4a29651191ad832f38b391f79df113a63e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/2c7e7296d8ec1910977e69a3bb28e5cc4dedc95c9df7597998932af816b55eda.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/4109c9dbe59ecb7971afef41d33dc0868db4f7055c244c8d08881cb1bd9eab39.jpg)



Time step



Figure 9. Initial time series and mapping images.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/7ac0ae2d9cbe0551c6a442edb3ba00db659ab97b396992bf51734ca2e75821fc.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/7777420b27eaba24e2eab2214d57267933e73efb30a78490c6e798a171837aa5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/8ce36f6a2cfbb1e3b8a4ea440765c36b54f772528b8cde438c6c7dce0b1ee274.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/97151141cf40719fe7e125b2343ccc1e9a84f1e466e9aded1dfb641323a1c7a2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/824dae531a91512d2146c0f7e4439dd26b7074278b2cf7d795bda6e8286d2999.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/894dbf29317a22924376d87bce856082bec1290d9a5fb4e4eb02244a94ad0ee0.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-05-26/26cdbace-5323-41dc-930f-287969b94d37/83f18f8826011a810b136e2d732aafa90e9974570d74818fcbc125a7e99b2c55.jpg)



Time Step



Figure 10. Qualitative visualization of anomaly detection results. Blue curve: original time series; red markers: ground-truth anomaly labels; green curve: predicted anomaly scores.


# C.4. Comparison under different anomaly types

To demonstrate the effectiveness of our method on both global and local anomalies, we categorized and reorganized the NAB and YAHOO datasets into local, global, and mixed anomaly types. Our approach is compared with VIT4TS and TimeRCD, and the results are presented in Table 10. By synergizing the fine-grained sensitivity of temporal features with the global semantic context of visual representations, VETime effectively mitigates the limitations of unimodal baselines and achieve consistent superiority across all anomaly categories. 

# D. Visualized results

# D.1. Visualized mappings

Figure 9 visualizes the representations generated by our Reversible Image Conversion module across diverse data samples. The left panels display the raw 1D time-series sequences with ground-truth anomalies highlighted in red, while the right panels present their corresponding 2D visual transformations. As observed, the generated images effectively encode key temporal characteristics, such as periodicity, trends, and noise, into distinct visual textures and patterns. This visualization confirms that our conversion method successfully preserves information-dense global contexts, providing a discriminative basis for the subsequent multi-modal fusion. 

# D.2. Visualized anomaly scores

Figure 10 presents qualitative anomaly detection results across heterogeneous time series patterns, validating the method’s robustness to diverse anomaly patterns. For point anomalies—such as abrupt spikes or isolated dips (visible in Rows 1–2), the green anomaly score curve exhibits sharp, well-localized peaks that precisely align with annotated labels. For context anomalies characterized by extended deviations like sustained trend shifts or contextual drifts (Row 3), the anomaly scores maintain elevated levels throughout the entire anomalous interval, maintaining continuous and coherent alignment with the red-labeled events. 