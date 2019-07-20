Unsupervised Bin-wise Pre-training: A Fusion of Information Theoretic and

Hypergraph Concepts

Abstract:
Minimizing the training time of the Deep Neural Networks still remains a significant
challenge, as the parameters are huge. There arises the need for optimizing and
regularizing the parameters within minimal time. To achieve this, pre-training is one of
the promising technique which is widely preferred by the researchers and considered to
be a ‘starting point’ of the Deep Neural Networks. Plenty of pre-training models are
presented in the recent research works however they often fail to capture the relevant
information and to maintain the stability of the learning model. Hence, this research
article presents a novel unsupervised bin-wise pre-training model which fuses
Information Theory, Partial Information Decomposition and Hypergraph concepts to
speed up the learning process and to minimize the training &amp; validation loss of the Deep
Neural Networks through improved feature representation. Further, a new approach of
parameter updation during pre-training has been introduced that acts both as an
optimizer &amp; a regularizer. The proposed model has been evaluated using MNIST
benchmark image dataset and the experimental results confirm the effectiveness of the
proposed unsupervised bin-wise pre-training model in terms of regularization &amp;
optimization capability and achieves competitive results compared to the state-of-the-
art approaches.
Keywords: Deep Neural Network; Mutual Information; Information Theory; Partial
Information Decomposition; Hypergraph;
1. Introduction:
Deep Neural Network (DNN) - an integral part of Deep Learning (DL) guarantees higher
accuracy and flexibility by learning to represent “the world as a nested hierarchy of
Information, with each defined in relation to simpler ones” [1] . Powerful features of
DNN such as an increase in robustness and performance as the data increases, learning
higher-level features from the data incrementally without feature engineering, end-to-
end problem-solving capability, etc., make four among five researchers believe that the
advent of DNN makes life easier [2] . However, parameter initialization is one of the
major issues of DNN as it affects the rate of convergence and the generalization of the
model [3–10] .
To address this issue, pre-training is widely adapted in DNN as it helps in finding a
better starting point in loss topology for improved Empirical Risk Minimization [11] .
Pre-training is a process of adding new hidden layers for constructing a DL model and
permitting the newly added layer to acquire the information from the preceding hidden
layers [12] . Predominantly, Unsupervised Pre-training focuses on weight updation for
effective feature transformation and representation through layers, which reduces the
high time-consuming exploration phase of the Optimization algorithm [13] . Among the
existing unsupervised pre-training approaches Deep Belief Networks (DBN),
Autoencoders and its variants are extensively used for pre-training [14] . However, the
existing pre-training strategies suffer due to computational complexity and perform
compression rather conceptualization.

On the other side, recent research works on understanding unfathomable concepts of
DNN [15–20] to improve state-of-the-art methods through Information Theory has
proven successful. Information Theory discloses how parameters are motivated to
acquire the information from the known data and plausibly able to expound the trends
observed during training [20] . This different perspective of viewing DNN helps us to
answer, how model proceeds to optimize instead of a stochastic step. However, a very
few have attempted to use Information Theory to solve the limitations of pre-training
and yet they lack to exploit the complete significance of Information Theory.
To overcome the aforementioned issues, this paper proposes a novel pre-training model
which fuses Information Theory, Partial Information Decomposition (PID) and
Hypergraph concepts in an unsupervised fashion. The novelty and major contributions
of the proposed unsupervised bin-wise pre-training model are as follows:
1. Mutual Information and PID based Hypergraph construction has been proposed
for pre-training
2. A novel parameter updation during pre-training has been introduced that
performs both optimization &amp; regularization and the proper justifications are
provided
3. The -helly property of hypergraph has been employed as a stopping criterion for
pre-training
4. The MNIST benchmark dataset was used to evaluate the proposed model and the
results have been compared with the traditional weight initialization techniques
&amp; other existing pre-training models to confirm the predominance of the
proposed pre-training model
The rest of the paper is structured as follows: Section 1 discusses the background and
importance of the proposed model with its novelty and contributions. Section 2 briefly
portrays the recent and the traditional practices along with their pitfalls. Section 3
describes the necessary preambles of the proposed model. Section 4 reveals the
importance and the intuition behind the proposed unsupervised bin-wise pre-training
model. Section 5 deals with the experimental setup and presents several aids for
showing the supremacy of the proposed model. Section 6 concludes the article along
with the applications and the future direction.