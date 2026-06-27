# GuidedQuant Paper Notes

Source PDF: `/Users/nguyenquanghuy/Downloads/guide.pdf`

This Markdown file is a repo-local context copy for coding/reproduction work. The first section below is hand-curated for implementation decisions; the later section is text extracted from the PDF content streams and may contain layout/OCR artifacts.

## Coding Context

- Paper: **GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance**.
- Core idea: incorporate gradient information from the end loss into layer-wise PTQ objectives while preserving cross-weight dependencies within output channels.
- GuidedQuant requires a calibration backward pass. The algorithm stores averaged squared gradients and then runs the layer-wise quantizer. This matters for memory: full-sequence backward is the expensive part.
- Main experimental calibration setup: **RedPajama**, **1024 sentences**, **4096 tokens each**.
- Main PPL evaluation setup: WikiText2 and C4 validation perplexity, measured with **context size 4096**.
- Main weight-only scalar comparison includes **SqueezeLLM without mixed precision** as a baseline. Do not confuse this with the SqueezeLLM dense-and-sparse / mixed-precision variant.
- Appendix dense-and-sparse comparison follows original SqueezeLLM style and retains **0.45%** of weights in 16-bit.
- When using our C4 reproduction scripts, remember that current low-memory default is **C4 128 samples x 2048 tokens**, which is intentionally not the paper's RedPajama 1024 x 4096 setup.
- For exact paper-like SqueezeLLM baseline, prefer RedPajama 1024 x 4096 when hardware permits. For A40-class single GPU, C4 128 x 2048 is a practical debugging/ablation config.

## Key Results/Setup Snippets

- Table 3 caption: weight-only scalar post-training quantization results without fine-tuning with end-to-end loss; Wiki2 and C4 denote perplexity on WikiText2 and C4; context size is 4096.
- Llama-2 SqueezeLLM rows in Table 3 include 2-bit, 3-bit, and 4-bit baselines. At 2-bit, SqueezeLLM is weak on Llama-2-7B compared with LNQ + GuidedQuant; at 3/4-bit it is strong.
- Paper setup text states calibration data follows prior work and uses RedPajama with 1024 sentences of 4096 tokens.

## Extracted PDF Text

GuidedQuant: Large Language Model Quantization viaExploiting End Loss Guidance
Jinuk Kim1 2 yMarwa El Halabi3Wonpyo Park4Clemens JS Schaefer4Deokjae Lee1 2Yeonhong Park1 Jae W. Lee1Hyun Oh Song1 2
Abstract
Post-training quantization is a key technique forreducing the memory and inference latency oflarge language models by quantizing weights andactivations without requiring retraining. How-ever, existing methods either (1) fail to accountfor the varying importance of hidden features tothe end loss or, when incorporating end loss, (2)neglect the critical interactions between modelweights. To address these limitations, we proposeGuidedQuant, a novel quantization approach thatintegrates gradient information from the end lossinto the quantization objective while preservingcross-weight dependencies within output chan-nels. GuidedQuant consistently boosts the per-formance of state-of-the-art quantization methodsacross weight-only scalar, weight-only vector, andweight-and-activation quantization. Additionally,we introduce a novel non-uniform scalar quantiza-tion algorithm, which is guaranteed to monoton-ically decrease the quantization objective value,and outperforms existing methods in this category.We release the code at
https://github. com/snu-mllab/GuidedQuant
.1. IntroductionLarge language models (LLMs) have shown remarkablecapabilities across a range of tasks, from text generation tocomplex reasoning. However, these advancements come atthe cost of substantial memory usage and inference latency.Quantization provides an effective solution to these chal-lenges. Weight-only quantization methods quantize only the
yWork partly done during an internship at Google.1Departmentof Computer Science and Engineering, Seoul National Univer-sity2Neural Processing Research Center3Samsung AI Lab,Montreal4Google. Correspondence to: Hyun Oh Song<hyunoh@snu.ac.kr>.Proceedings of the42ndInternational Conference on MachineLearning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025by the author(s).
Table 1.
Summary of results of GuidedQuant applied to state-of-the-art PTQ methods on the Llama-2-7B model. Wiki2-4K andWiki2-2K represent perplexity on WikiText2 dataset with con-text size of 4096 and 2048, respectively. W4A4KV4 indicatesquantization of all weight, activation, and KV cache to 4 bits.
Method Bits# Wiki2-4K#
Type Original 16 5.12
Weight-onlyScalarSqueezeLLM 2.01 39.58LNQ (Ours) 2.01 23.31LNQ + GQuant (Ours) 2.01 8.83
Weight-onlyVectorQTIP 2.00 6.82QTIP + GQuant (Ours) 2.00 6.11
Method Bits# Wiki2-2K#
Type Original 16 5.47
Weight-and-ActivationSpinQuant W4A4KV4 5.95SpinQuant + GQuant (Ours) W4A4KV4 5.89
model weights, reducing data transfer and thus acceleratinginference in memory-bound scenarios such as small-batchinference (
Gholami et al.
,
2024
;
Kim et al.
,
2024
;
Tsenget al.
,
2024b
). On the other hand, weight-and-activationquantization methods quantize both the model weightsand activations. In addition to reducing data transfer, thesemethods also speed up arithmetic operations, making themparticularly beneficial for large-batch scenarios such aspre-filling input tokens or generating batched samples(
Ashkboos et al.
,
2024
;
Liu et al.
,
2024
). Weight-onlyquantization techniques have used three grid types: uniformscalar (
Frantar et al.
,
2023
), non-uniform scalar (
Kim et al.
,
2024
), and vector quantization (
Tseng et al.
,
2024b
;
vanBaalen et al.
,
2024
), each with its own advantages (seeSection
5
for details). In contrast, weight-and-activationmethods typically use a uniform scalar grid, as using anon-uniform grid would require dequantization before multi-plication, preventing the use of faster arithmetic operations.Quantization benefits come at the cost of performance degra-dation. Quantization-Aware Training (QAT) methods relyon retraining the quantized model to mitigate this, whichis prohibitively expensive at the scale of modern LLMs. Inconstrast, Post-Training Quantization (PTQ) methods quan-tize the pretrained model using a small calibration dataset or
1
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
@`
@(XW)2 Rnfidout
8j 2 f1; : : : ; doutg :
Hj= X>Diag(@`
@zj)2X
0B@
1CA>
cW:;j
W:;j
Hj
0B@
1CA
cW:;j
W:;j
8k 2 f1; : : : ; gg : (g dout)
Hk=1
jJkjPj2JkHj
0B@
1CA>
cW:;Jk
W:;Jk
Hk
0B@
1CA
cW:;Jk
W:;Jk
X 2 Rnfidin
0BB@
1CCA
0BBBB@
1CCCCA
2F
W 2 Rdinfidout
cW 2 Rdinfidout
J1
: : :
Jg
J1
: : :
Jg
Figure 1.
Top: The proposed GuidedQuant's layer-wise quantization objective(
4
). Bottom-left: Its equivalent quadratic form(
6
).Bottom-right: The approximated objective(
7
)proposed in Section
3.2
. We denote the input, weight, and quantized weight matrices asX 2 Rnfidin,W 2 Rdinfidout, and^W 2 Rdinfidout, respectively. The groupsJ1; : : : ; Jgform a partition of the setf1; : : : ; doutg, andzj2 Rdoutdenotes the j-th column of Z = XW.
no data, without retraining the entire model. Most existingPTQ methods for LLMs rely on a surrogate objective ratherthan the end loss to make quantization feasible.One common PTQ strategy, which we refer to as layer-wise output-based quantization, aims to quantize each layerby minimizing the mean squared error between the layer'soriginal output and the quantized one (
Nagel et al.
,
2020
;
Frantar et al.
,
2023
;
Egiazarian et al.
,
2024
;
Chee et al.
,
2024
;
Tseng et al.
,
2024a
;
b
;
Liu et al.
,
2024
). However, thisstrategy treats all hidden features equally, overlooking theirvarying impact on the end loss.Alternatively, methods such as
Choi et al.
(
2017
);
Kim et al.
(
2024
) leverage gradient information from the end loss toassess the impact of individual weight errors. This is doneby computing the gradient of the end loss with respect toweights via a single backpropagation step on a calibrationdataset. Saliency scores are then assigned to weights basedon these gradients, and the model is quantized by approx-imately minimizing the sum of saliency-weighted weighterrors. This objective corresponds to a quadratic approxima-tion of the change in the end loss, based on its second-orderTaylor expansion, where the Hessian is approximated by thediagonal of the empirical Fisher information matrix (
Has-sibi & Stork
,
1992
). A key limitation of this approach isthat it ignores cross-weight interactions, which are crucialfor overall performance.Contributions In this work, we propose GuidedQuant,a novel PTQ approach that integrates gradient informationfrom the end loss while preserving cross-weight dependen-cies within output channels. In particular, GuidedQuant
computes saliency scores for layer outputs using the gra-dients of the end loss with respect to these outputs. Eachlayer is then quantized independently by approximately min-imizing the sum of saliency-weighted output errors. Unlikeprevious methods that assume a diagonal Hessian, this ob-jective is equivalent to a refined quadratic approximationassuming a block-diagonal Hessian, again approximated bythe empirical Fisher information matrix. While cross-layerand cross-output channel interactions are still ignored, de-pendencies within output channels are preserved, enablinga more accurate estimation of quantization's impact on theend loss.Computing and storing the diagonal blocks of the Fishermatrix for a given layer is too expensive for modern LLMs.To address this, we partition the layer's outputs into a smallnumber of groups and average the Fisher matrix's blockswithin each group (Figure
1
). Other block-diagonal Fishermatrix approximations of the Hessian have been used forpruning CNNs (
Singh & Alistarh
,
2020
) and BERT LLMs(
Kurtic et al.
,
2022
) with arbitrary blocks along the diagonal,and for quantizing CNNs (
Li et al.
,
2021
) with diagonalblocks corresponding to the model's residual blocks (seeSection
D.11
for more details). However, our work is thefirst to make this approach computationally and storage-efficient at the scale of modern LLMs.GuidedQuant can be applied as a direct plug-in to any layer-wise output-based PTQ method. We demonstrate its effec-tiveness by integrating it into the current state-of-the-artmethods for weight-only vector quantization, QTIP (
Tsenget al.
,
2024b
), and weight-and-activation quantization, Spin-Quant (
Liu et al.
,
2024
), which are both layer-wise output-
2
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
based PTQ methods. GuidedQuant consistently improvestheir performance (Table
1
).For weight-only scalar quantization, the current state-of-the-art methods are SqueezeLLM (
Kim et al.
,
2024
) andGPTVQ 1D (
van Baalen et al.
,
2024
). Since GPTVQ 1Dis a layer-wise output-based PTQ method, GuidedQuantcan be applied to it. However, GPTVQ 1D employs asuboptimal algorithm for minimizing layer-wise output er-rors. To address this, we introduce a novel Layer-wiseNon-uniform Quantization method, LNQ, which minimizelayer-wise output errors using an alternating minimizationalgorithm, where the codebook is optimized in closed-form,and assignments are optimized via a coordinate descent(CD) algorithm. LNQ outperforms GPTVQ 1D and matchesor surpasses SqueezeLLM. Applying GuidedQuant to LNQfurther improves its performance, achieving state-of-the-artresults (Table
1
).2. PreliminariesConsider a neural network withLlinear layers, trained witha loss function`and a calibration data of sizen. We de-note the loss computed on thei-th data point as`i. LetW(l)2 Rd(l)infid(l)outbe the weight matrix of thel-th linearlayer, where each column vectorw(l)j2 Rd(l)incorrespondsto an output channel. We denote its quantized approximationascW(l). The input and output feature maps of this layer areX(l)2 Rnfid(l)inandZ(l)2 Rnfid(l)out, respectively. The out-put of the linear layer is computed asZ(l)= X(l)W(l),and the output after quantization asbZ(l)= X(l)cW(l).Letw = [vec(W(1))>; ; vec(W(L))>]>andbw = [vec(cW(1))>; ; vec(cW(L))>]>be the vectors ofweights in allLlayers before and after quantization, wherevec(W`) corresponds to stacking the columns of W`.Most existing PTQ methods for LLMs are layer-wise output-based quantization methods, which quantize each layer byapproximately minimizing the objectivekX(l)W(l) X(l)cW(l)k2F=nXi=1d(l)outXj=1Z(l)ijbZ(l)ij2(1)ignoring the varying impact of outputs on the end loss`.Existing methods employ various heuristics to minimize thisobjective, such as AdaRound (
Nagel et al.
,
2020
), CD meth-ods (
Nair & Suggala
,
2024
;
Behdin et al.
,
2023
;
Egiazarianet al.
,
2024
;
Chee et al.
,
2024
), OBQ (
Frantar & Alistarh
,
2022
), GPTQ
1
(
Frantar et al.
,
2023
), GPTVQ (
van Baalenet al.
,
2024
), and AQLM (
Egiazarian et al.
,
2024
).A more accurate proxy objective, first introduced in earlypruning methods (
LeCun et al.
,
1989
;
Hassibi & Stork
,
1Also referred to as OPTQ.
1992
), is the following quadratic approximation of thechange in the end loss`(bw) `(w) 1
2(bw w)>r2`(w)(bw w): (2)This approximation is derived from the second-order Taylorapproximation of`, assuming that the trained model hasconverged and thus the gradient is close to zero. Since com-puting the Hessian is infeasible even for small models, a pop-ular approach first proposed in
Hassibi & Stork
(
1992
) ap-proximates the Hessian by the empirical Fisher informationmatrixF =1
nPni=1r`i(w)r`i(w)>, which yields thefollowing quadratic approximation, (bw w)>F(bw w):SqueezeLLM (
Kim et al.
,
2024
) is a weight-only non-uniform scalar PTQ method for LLMs which uses thisquadratic approximation, but further approximates theFisher information matrix by its diagonaldiag(F), ignoringoff-diagonal entries. The resulting objective is given by(bw w)>diag(F)(bw w) =XkFkk( bwk wk)2: (3)For non-uniform scalar quantization, minimizing this ob-jective corresponds to solving a weightedk-means problemin 1D, which can be solved exactly using a dynamic pro-gramming algorithm (
Grønlund et al.
,
2017
). SqueezeLLMinstead employs Lloyd's algorithm with k-means++ initial-ization (
Lloyd
,
1982
;
Arthur & Vassilvitskii
,
2007
), whichis only guaranteed to achieve afi(log k)approximation inexpectation, wherekis the number of clusters, but is fasterin practice (
Hyun
,
2024
). However, the diagonal approxi-mation is highly inaccurate, as both the Hessian matrix andits Fisher approximation are usually strongly non-diagonal,as observed in prior work for small CNNs (
Hassibi & Stork
,
1992
;
Singh & Alistarh
,
2020
). We also confirm this observa-tion for the Fisher matrix of Llama-2-7B in Figures
3
and
4
.3. GuidedQuantIn this section, we introduce our PTQ approach Guid-edQuant. We first propose a layer-wise quantization ob-jective that more accurately approximates the impact ofquantization on the final loss compared to surrogate objec-tives used in existing PTQ methods. We then present a sim-plified version of this objective, making it computationallyand memory efficient for LLMs with up to 70B parameters.3.1. ObjectiveAs discussed earlier, most existing PTQ methods treat alloutput features as equally important, by employing the sur-rogate objective in Eq.(
1
). In contrast, we propose tomodify this objective to account for the varying impact ofeach output feature on the final loss.To that end, we approximate the change in the end loss`resulting from the output featureZ(l)ijchanging tobZ(l)ijafter
3
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
2.0
3.0
4.0
5.0
5.5
6.0
8
24
40
Bits #
WikiText2 Perplexity #
Original-FP16
SqueezeLLM (weighted k-means)
LNQ (layer-wise)
LNQ + GuidedQuant (our objective)
Figure 2.
Non-uniform scalar quantization results on Llama-2-7Bwith different objectives: layer-wise output error objective(
1
)usedin LNQ (Algorithm
2
), weightedk-means objective(
3
)used inSqueezeLLM, and our approximated GuidedQuant objective(
7
)used in LNQ combined with GuidedQuant. We report perplexity onWikiText2 with a context size of 4096. Results are from Table
3
.
quantization, using a first-order Taylor expansion, assumingindependence of output features:`(bZ(l)ij) `(Z(l)ij) @`
@Z(l)ij(bZ(l)ij Z(l)ij):Accordingly, we propose to scale each output error by thegradient of the end loss with respect to that output, leadingto the following layer-wise objective:@`
@Z(l) (X(l)W(l) X(l)cW(l))2F =nXi=1d(l)outXj=1 @`
@Z(l)ij(Z(l)ijbZ(l)ij)!2; (4)wheredenotes the element-wise multiplication. Thiscriterion was previously proposed in
Molchanov et al.
(
2019
)for pruning neurons and filters in vision models, wherepruning thejth neuron in layerlcorresponds to settingbw(l)j= 0.We note that the objective in Eq.(
4
)can be viewed as a sim-plification of the second-order Taylor approximation of thechange in the end loss given in Eq.(
2
), where the Hessianis approximated by the empirical Fisher information matrix,and where interactions between weights belonging to differ-ent layers or output channels of the same layer are ignored.In other words, we adopt a block-diagonal approximationof the Fisher matrixFwhere we only keep thed(l)infi d(l)inblocksF(l)j=1
nPni=1(@`i
@w(l)j)(@`i
@w(l)j)>corresponding tointeractions within each output channeljof every layerl,and ignore all off-block entries.
Remark 3.1.
The sum of the layer-wise objective in Eq.(
4
)over all layers is equal to the following quadratic approxi-mation of the change in the end lossnLXl=1d(l)outXj=1(w(l)jbw(l)j)>F(l)j(w(l)jbw(l)j): (5)The proof of Remark
3.1
follows from the chain rule, andis given in Section
A
. A similar observation was made in
Molchanov et al.
(
2019
).Assuming that the quantization grid used is separable overlayers, which is typically the case, minimizing the objectivein Eq. (
5
) is equivalent to independently minimizingd(l)outXj=1(w(l)jbw(l)j)>H(l)j(w(l)jbw(l)j); (6)for every layer, whereH(l)j= nF(l)j, or equivalently thelayer-wise objective in Eq. (
4
).Thus our proposed objective is a more accurate approxima-tion of the change in the end loss than the layer-wise outputerror objective(
1
), which assumes@`
@Z(l)/ I, as well asthe weightedk-means objective(
3
)used in SqueezeLLM,which ignores all off-diagonal entries in the Fisher matrixincluding those within the blocksF(l)j. As a result, ourapproach achieves better performance, even with the ad-ditional approximation discussed in Section
3.2
, as high-lighted in Figure
2
for non-uniform scalar quantization, andlater across other formats in Section
5
.In Figures
3
and
4
, we visualize a submatrix of the Fisherinformation matrix corresponding to the first two outputchannels in the linear layers of the first Transformer blockof Llama-2-7B. The visualization confirms that the Fishermatrix exhibits strong off-diagonal values and a prominentblock-diagonal structure, with blocks corresponding toF(l)j for the two output channels j 2 f1; 2g.3.2. Averaging ApproximationThe layer-wise output error objective (
1
) can be written asd(l)outXj=1w(l)jbw(l)j>H(l)w(l)jbw(l)j;whereH(l)= X(l)>X(l)2 Rd(l)infid(l)in. Most existingheuristics for optimizing this objective, such as GPTQ (
Fran-tar et al.
,
2023
) and CD (
Nair & Suggala
,
2024
;
Behdinet al.
,
2023
), require access only toH(l)and notX(l).Thus, the Hessian matrixH(l)is typically precomputed,which reduces the peak memory usage during optimiza-tion, sinceX(l)2 Rnfid(l)inis much larger thanH(l), given
4
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
thatd(l)in n. Additionally, the precomputed Hessian canbe reused across multiple quantization configurations andbit-widths, amortizing the cost of its computation.Our proposed objective(
6
)can be seamlessly integratedinto any layer-wise output based quantization method byreplacingH(l)byH(l)j= nF(l)jfor each output channelj.However, precomputing and storingH(l)jfor alljincurs amemory cost offi((d(l)in)2d(l)out)and a time complexity offi(n(d(l)in)2d(l)out)per layerl. This is infeasible at the scaleof modern LLMs, where bothd(l)inandd(l)outexceed103, andn is much larger than both.To address this challenge, we partition the output channelsof each layer intogdistinct groups (g d(l)out) and replacethe individual Hessian matricesH(l)jwithin each groupkby a shared matrix
H(l)k, obtained by averagingH(l)jwithinthe group. Formally, letJ(l)1; : : : ; J(l)gbe a partition ofthe setf1; : : : ; d(l)outg. For each groupk = 1; : : : ; g, wedefine
H(l)k=1
jJ(l)kjPj2J(l)kH(l)j:The resulting layer-wiseobjective then becomesgXk=1Xj2J(l)kw(l)jbw(l)j>
H(l)kw(l)jbw(l)j: (7) Note that by the chain rule, we can writeH(l)j= X(l)>Diag @`
@z(l)j!2X(l);whereDiag(@`
@z(l)j)2is the diagonal matrix whose diagonalentries are the element-wise square of the gradient of`withrespect to thejth columnz(l)jofZ(l). We can thus compute
H(l)kby averaging the squared gradients:
H(l)k= X(l)>Diag0@1
jJkjXj2Jk @`
@z(l)j!21AX(l):This averaging approximation reduces the number ofd(l)infi d(l)inHessian matrices that need to be computed for eachlayerlfromd(l)outtog(Figure
1
). Computing and storing
H(l)kfor allkrequires a significantly lower memory costoffi((d(l)in)2g)and time complexity offi(n(d(l)in)2g)perlayerl(assuming the squared gradients averages are alreadycomputed), making the method scalable. To partition theoutput channels, we use a simple strategy that groups ev-eryd(l)out=gconsecutive channels into a single group. Thissimple approach works well in practice, though more sophis-ticated clustering algorithms may yield additional benefits.
Algorithm 1 GuidedQuant
input
Layer-wise quantization algorithmQ, number ofgroups g, number of linear layers L
1:
J(l)k fd(l)out
g(k 1) + 1; : : : ;d(l)out
gkg; 8l 2 [L]; k 2 [g]
2:
s(l)k 1
jJkjPj2Jk(@`
@z(l)j)2; 8l 2 [L]; k 2 [g]
3:
for all l 2 [L]; k 2 [g] do
4:
H(l)k X(l)>Diag(s(l)k)X(l)
5:
cW(l)h:; J(l)ki Q
H(l)k; W(l)h:; J(l)ki
6:
end for
output
cW(1); : : : ;cW(L).
In our implementation, we scale the gradients by a largeconstant (we used103in all experiments) while computingthe averaged Hessians
Hkto prevent underflow.GuidedQuant quantizes each layer independently by ap-proximately minimizing the layer-wise objective in Eq.(
7
).A complete overview of GuidedQuant is provided in Algo-rithm
1
. As discussed, the layer-wise quantization algorithmQcan be any layer-wise output based quantization method.The gradient computation (Line 2) requires a single back-propagation step on the calibration dataset. During this step,we only store the averaged squared gradientss(l)k, whichrequires O(ngL) storage.The total memory cost of GuidedQuant (without the back-propagation step) is thenO(Lg(d2in+ n)), and its total timecomplexity isOLg(nd2in+ TQ(din; dout=g)), wheredin; doutare the largest input and output channel dimensionsacross allLlayers andTQ(d1; d2)is the time complexity ofquantizing ad1fi d2-weight matrix usingQ. Each step inthe for loop (Lines 3-6) can be done in parallel for all groupsand layers. As previously discussed, the Hessian matrices
Hk's only need to be computed once, and can be reused fordifferent quantization configurations and bit-widths.4. Layer-wise Non-uniform QuantizationThe choice of the layer-wise output based quantizationmethodQin GuidedQuant is critical to its overall perfor-mance. For weight-only non-uniform scalar quantization,the current state-of-the-art layer-wise output based quan-tization method is the 1D variant of GPTVQ (
van Baalenet al.
,
2024
), which alternates between optimizing the code-book via gradient descent and the assignments via GPTQalgorithm (
Frantar et al.
,
2023
). However, both of thesesteps can be improved. Given fixed assignments, the code-book admits an optimal closed form solution. Also, foroptimizing assignments, recent works have demonstratedthat coordinate descent (CD) methods outperform GPTQ inuniform weight-only quantization (
Behdin et al.
,
2023
;
Nair& Suggala
,
2024
). In this section, we introduce Layer-wise
5
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Algorithm 2 LNQ
input
Hessian of the objectiveH 2 Rdinfidin, input weightW 2 Rdinfidout, initial assignmentP(j)2 Rdinfimforeach output channel j.
1:
H = LL>fCholesky decompositiong
2:
for j 2 f1; : : : ; doutg do
3:
for t = 1 to T do
4:
c(j) P(j)>LL>P(j)1P(j)>LL>wj
5:
^wj P(j)c(j)
6:
for k = 1 to K do
7:
for i = 1 to dindo
8:
c(j)qfl argmincWij2fc(j)1;:::;c(j)mg(bwjwj)>H(bwjwj)
9:
cWij c(j)qfl
10:
8q 2 f1; : : : ; mg : P(j)iq=(1 if q = qfl;0 otherwise.
11:
end for
12:
end for
13:
end for
14:
c(j) P(j)>LL>P(j)1P(j)>LL>wj
15:
end for
output
cW = [P(1)c(1); : : : ; P(dout)c(dout)].
Non-uniform Quantization (LNQ), an alternating minimiza-tion algorithm which leverages the closed form solutionfor the codebook and employs CD to optimize the assign-ments. We then discuss its theoretical guarantees, as well asthe memory cost and computational complexity under ourefficient implementation.4.1. Optimization ProblemWe omit the layer indexlfor notational simplicity through-out this section. Following prior work, we assign to eachoutput channel a separate codebook, though LNQ can beeasily adapted to finer-granularity grouping. Non-uniformscalar quantization maps each scalar weight in the columnwj2 Rdinto one ofm = 2breal valuesfc(j)1; : : : ; c(j)mg,whereb 2 Nis the target bit-width. The quantizedweightsbwjcan then be expressed asbwj= P(j)c(j), wherec(j)2 Rmis the vector containing the codebook valuesfc(j)1; : : : ; c(j)mg, andP(j)2 f0; 1gdinfimis the assignmentmatrix such thatP(j)iq= 1ifWijis assigned toc(j)q, andP(j)iq= 0 otherwise.The optimization problem for layer-wise output-based non-uniform scalar quantization can then be written as follows:minimizeP(j)2f0;1gdinfimc(j)2RmdoutXj=1kXwj XP(j)c(j)k22 subject to P(j)1m= 1din; (8)
where1is the vector of all ones. Note that the optimizationfor each columnjis independent of other columns, and canbe done in parallel.4.2. LNQ AlgorithmWe propose LNQ, an alternating minimization algorithm,which iteratively updates the codebookc(j)and assignmentmatrixP(j)for eachj, optimizing one while keeping theother fixed. Alternating minimization is a common strategyused by most non-uniform quantization methods, includingSqueezeLLM and GPTVQ. LNQ quantizes each layer inde-pendently. We present an overview of LNQ, applied to onelayer with weights W 2 Rdinfidoutin Algorithm
2
.Given fixed assignment matricesP(j), Problem(
8
)reducesto a standard least-squares problem, which admits a closed-form optimal solutionc(j)fl= (XP(j))yXwj, whereydenotes the MoorePenrose pseudoinverse. We assume thatthe matrixP(j)>HP(j)is invertible, where recall thatH = X>X. Under this assumption, the closed-form solution is:c(j)fl=P(j)>HP(j)1P(j)>Hwj: (9)In practice,P(j)>HP(j)is not always invertible, even whenHis invertible (for example if no weight is assigned to agiven codebook valuec(j)q). To address this, we add a smallconstant = 107to the diagonal of the matrix, as com-monly done in prior work (
Frantar & Alistarh
,
2022
;
Frantaret al.
,
2023
;
van Baalen et al.
,
2024
). In our implementation,we usetorch.linalg.lstsqfunction to compute theleast squares solution in Eq.(
9
), which takesXP(j)andXwjas inputs. However, sinceXis not explicitly stored,we compute the Cholesky decomposition ofH = X>X, de-noted asH = LL>, and instead provideL>P(j)andLwjto the solver. Because Cholesky decomposition requiresHto be positive definite, we ensure this by adding a smallconstant to the diagonal of H.For fixed codebooksc(j), Problem(
8
)can be equivalentlywritten asminimizebwj2fc(j)1;:::;c(j)mgdindoutXj=1(bwj wj)>H(bwj wj): (10)Even in the special case of uniform codebook, this problemcorresponds to a closest vector problem with box constraints,which is NP-Hard to approximate within any constant factorapproximation form 2(
Arora et al.
,
1997
, Theorem1). Existing heuristics for solving it include OBQ (
Fran-tar & Alistarh
,
2022
) which does not scale to LLMs withbillions of parameters; its faster variant GPTQ (
Frantaret al.
,
2023
); LDLQ (
Chee et al.
,
2024
), which is a moreefficient implementation of GPTQ; greedy CD (
Nair & Sug-gala
,
2024
); and cyclic CD (
Behdin et al.
,
2023
;
Chee et al.
,
6
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
2024
;
Egiazarian et al.
,
2024
). Recent works show thatboth greedy CD (
Nair & Suggala
,
2024
) and cyclic CD(
Behdin et al.
,
2023
) outperform GPTQ on this problemwhen using a uniform grid. We thus adopt the cyclic CDalgorithm, since it performs similarly to the greedy variantwhile being significantly less expensive (
Nair & Suggala
,
2024
, Appendix D). In Section
D.6
, we present an ablationstudy that further support this choice, showing that cyclicCD matches or outperforms GPTQ when used within LNQfor non-uniform scalar quantization.Cyclic CD is an iterative algorithm which iterates over coor-dinates in a fixed order, minimizing at each iteration the ob-jective with respect to one coordinate, while keeping all oth-ers fixed. The minimization for each coordinate (Line 8 inAlgorithm
2
) has a closed form solution, as shown in
Behdinet al.
(
2023
, Lemma 1) and
Chee et al.
(
2024
, Section B.2):RoundjWi;jHi;[din]ni
Hi;i(cW[din]ni;j W[din]ni;j);(11)whereRoundj()denotes rounding to the nearest point inthe grid fc(j)1; : : : ; c(j)mg.CD is a descent method when initialized with a feasiblesolutionbwj2 fc(j)1; : : : ; c(j)mgdin, i.e., it monotonicallydecreases the objective function value. It can be used asa standalone solver for problem(
10
)initialized with theoriginal weightsW, as in
Nair & Suggala
(
2024
);
Behdinet al.
(
2023
), or to refine the output of another quantizationmethod, as done in the uniform quantization method QuIP,which runs CD after LDLQ (
Chee et al.
,
2024
).In LNQ, at each iteration, we initialize CD with the quan-tized weights corresponding to the current assignment andcodebook^wj= P(j)c(j)for eachj. For the first iter-ation, any feasible assignment matrix can be used. Inour experiments, we initialize with the assignments fromSqueezeLLM. Since the codebooks are updated optimallyand CD acts as descent method with feasible initialization, itfollows that LNQ itself is a descent method and it converges.Refer to Section
A
for the proof.
Proposition 4.1.
For anyj 2 [dout], letfj(c; P) = kXwj XPck22, and letc(j)tandP(j)tdenotec(j)andP(j)at thet-th iteration of LNQ. Then,fj(c(j)t; P(j)t) fj(c(j)t+1; P(j)t) fj(c(j)t+1; P(j)t+1)for allt, and the se-quence ffj(c(j)t; P(j)t)gt1converges.Since LNQ is a layer-wise output based method, Guid-edQuant can be easily applied to it. In Section
5.1
, wedemonstrate the efficacy of LNQ both as a standalone ap-proach and in combination with the GuidedQuant objective.Time Complexity. Computing the Cholesky decomposi-tion ofH(Line 1) requiresO(d3in), optimizing the code-
Table 2.
End-to-end inference throughput of Llama-2 models onRTX 4090 GPU. OOM indicates an Out-of-Memory error, meaningthe GPU lacks memory to run model inference. See Section
C.1
for experimental setup details.
Llama-2-7B Llama-2-13B Llama-2-70B
Type Bits# Tok/s" Bits# Tok/s" Bits# Tok/s"
Original 16 67 16 OOM 16 OOM
Uniform scalar 2.00 334 2.00 200 2.00 47Non-uniform scalar 2.01 347 2.01 203 2.01 47Vector 2.00 200 2.00 121 2.00 38
Uniform scalar 3.00 260 3.00 150 3.00 OOMNon-uniform scalar 3.03 264 3.02 148 3.01 OOMVector 3.00 176 3.00 103 3.00 OOM
Uniform scalar 4.00 214 4.00 121 4.00 OOMNon-uniform scalar 4.05 209 4.04 116 4.03 OOMVector 4.00 151 4.00 89 4.00 OOM
book (Line 4 and 14) requiresO(d2inm), and optimiz-ing the codes (Lines 6-12) requiresO(d2inK)time com-plexity. The total time complexity of LNQ algorithm isO(d3in+ d2indoutT (m + K)). Here,TandKdenotes thenumber of iterations for alternating optimization and thenumber of cycles in coordinate descent, respectively. Weprovide a detailed analysis of the time complexity in Sec-tion
B.2
. We discuss in Section
B.3
how to significantlyspeedup the implementation of CD on GPU, using precom-putation and lazy batch-updates. Precomputation is alsoused in
Behdin et al.
(
2023
);
Chee et al.
(
2024
), while lazybatch-updates is only used in
Chee et al.
(
2024
) (thoughnot discussed in the paper). These tricks do no change thetheoretical time complexity of CD, but they yield up to3fi speedups in practice.5. ExperimentsIn this section, we demonstrate the versatility and effective-ness of our method across various quantization schemes. Wefirst explore different quantization scenarios and identify theformats best suited to each setting, ultimately focusing onthree main approaches: weight-only scalar, weight-only vec-tor, and weight-and-activation quantization. By integratingthe GuidedQuant objective into existing methods, our re-sults consistently achieve state-of-the-art PTQ performance.Refer to Section
C.2
for details on how we incorporateGuidedQuant objective into existing methods. Additionalexperiments and details, including the overall cost of ourmethod, the effect of the number of groupsg, and the end-to-end fine-tuning results, are provided in Section
D
.5.1. Weight-only QuantizationExperimental Setup. Weight-only quantization primar-ily accelerates inference latency in low-batch scenarios,where memory bandwidth constitutes the main bottleneck
7
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Table 3.
Weight-only scalar post-training quantization results without fine-tuning with end-to-end loss. Wiki2 and C4 denotes perplexityon WikiText2 and C4, respectively. The perplexity is measured with the context size of 4096.
Llama-2-7B Llama-2-13B Llama-2-70B
Method Bits# Wiki2# C4# Bits# Wiki2# C4# Bits# Wiki2# C4#
Original 16 5.12 6.63 16 4.57 6.05 16 3.12 4.97
QuIP    2.00 13.48 16.16 2.01 5.90 8.17SqueezeLLM 2.01 39.58 44.05 2.01 16.24 19.20 2.01 9.17 13.03GPTVQ 1D 2.03 51.87 47.33 2.03 9.53 12.62 2.03 6.03 8.44LNQ (Ours) 2.01 23.31 26.71 2.01 8.78 11.80 2.01 5.23 7.31LNQ + GuidedQuant (Ours) 2.01 8.83 11.15 2.01 7.26 9.17 2.01 5.04 7.04
GPTQ 3.00 8.06 10.61 3.00 5.85 7.86 3.00 4.40 6.26QuIP    3.00 5.12 6.79 3.01 3.87 5.67SqueezeLLM 3.03 5.74 7.44 3.02 4.99 6.60 3.01 3.53 5.31GPTVQ 1D 3.03 6.17 8.02 3.03 5.13 6.76 3.03 3.55 5.35LNQ (Ours) 3.03 5.89 7.74 3.02 5.02 6.68 3.01 3.50 5.31LNQ + GuidedQuant (Ours) 3.03 5.57 7.22 3.02 4.91 6.49 3.01 3.47 5.27
GPTQ 4.00 5.49 7.20 4.00 4.78 6.34 4.00 3.35 5.15QuIP    4.00 4.76 6.29 4.00 3.58 5.38SqueezeLLM 4.05 5.23 6.78 4.04 4.67 6.15 4.03 3.20 5.04GPTVQ 1D 4.06 5.27 6.83 4.06 4.67 6.17 4.03 3.20 5.04LNQ (Ours) 4.05 5.26 6.82 4.04 4.67 6.17 4.03 3.20 5.04LNQ + GuidedQuant (Ours) 4.05 5.21 6.75 4.04 4.65 6.14 4.03 3.20 5.03
(
Gholami et al.
,
2024
). Among weight-only techniques,three quantization formats are commonly used: uniformscalar, non-uniform scalar, and vector quantization (
Fran-tar et al.
,
2023
;
Kim et al.
,
2024
;
Tseng et al.
,
2024b
). Withfixed bit-width constraints, non-uniform scalar quantiza-tion generally outperforms uniform scalar quantization, asits search space encompasses that of uniform scalar quan-tization. Meanwhile, vector quantization can outperformnon-uniform scalar quantization by exploiting additionalredundancies across weight dimensions.Despite this, non-uniform scalar quantization offers advan-tages in inference latency. Table
2
compares end-to-endsingle-batch inference latency across these formats usingthe state-of-the-art GPU kernels: LUT-GEMM (
Park et al.
,
2024a
) for uniform scalar, Any-Precision-LLM (
Park et al.
,
2024b
) for non-uniform scalar, and QTIP (
Tseng et al.
,
2024b
) for vector quantization. Results show that vec-tor quantization incurs higher latency due to its decodingoverhead (
Tseng et al.
,
2024b
), whereas uniform and non-uniform scalar quantization have similar latency with mini-mal decoding overhead. Consequently, non-uniform scalarand vector quantization remain the primary formats of inter-est for weight-only quantization. In this context, we applyour GuidedQuant to both formats, achieving state-of-the-artperformance in each.For our experiments, we demonstrate the effectiveness ofour method on the Llama-2 model family (
Touvron et al.
,
2023
), evaluating on 7B, 13B and 70B model. We usethe RedPajama dataset (
Computer
,
2023
) for calibration,
following prior work (
Egiazarian et al.
,
2024
;
Tseng et al.
,
2024a
;
b
), with 1024 sentences, each containing 4096 tokens.We report perplexity on the WikiText2 (
Merity et al.
,
2016
)and C4 (
Raffel et al.
,
2020
) validation sets.Scalar Post-training Quantization Results. We summa-rize the results of weight-only scalar quantization in Table
3
,comparing our approach with GPTQ (
Frantar et al.
,
2023
),SqueezeLLM without mixed precision (
Kim et al.
,
2024
),QuIP (
Chee et al.
,
2024
), and GPTVQ 1D (
van Baalen et al.
,
2024
). For GPTQ and QuIP, we report the results from
Egiazarian et al.
(
2024
), which used the same or a largercalibration dataset, while for GPTVQ 1D, we reproducethe results with the same calibration data while adjustingthe group size to align with the average bit-width for a faircomparison (see Section
B.4
for details).We evaluate the performance of LNQ both with and withoutthe GuidedQuant objective. Notably, LNQ combined withGuidedQuant consistently outperforms all baselines acrossvarious bit-widths and model sizes. Additionally, LNQ withthe layer-wise reconstruction objective surpasses GPTVQ1D in all settings, demonstrating that our approach improvesupon GPTVQ 1D by addressing its suboptimal optimization.Vector Post-training Quantization Results. For vectorpost-training quantization (PTQ), we present the results inTable
4
. We apply GuidedQuant to the state-of-the-art vectorPTQ baseline, QTIP (
Tseng et al.
,
2024b
). We implementit on both the 1MAD and 3INST variants and report thevariant that performs better among these two. Refer to
8
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Table 4.
Weight-only vector post-training quantization results without fine-tuning to the end-to-end loss. Wiki2 and C4 denotes perplexityon WikiText2 and C4, respectively. The perplexity is measured with the context size of 4096.
Llama-2-7B Llama-2-13B Llama-2-70B
Method Bits# Wiki2# C4# Bits# Wiki2# C4# Bits# Wiki2# C4#
Original 16 5.12 6.63 16 4.57 6.05 16 3.12 4.97
GPTVQ 2D 2.13 10.66 12.81 2.13 7.55 9.82 2.13 5.06 7.09GPTVQ 4D 2.25 7.89 10.25 2.25 6.36 8.43 2.25 4.44 6.28QuIP# 2.00 8.22 11.01 2.00 6.06 8.07 2.00 4.16 6.01AQLM 2.02 6.59 8.54 2.19 5.37 7.16 2.07 3.94 5.72QTIP 2.00 6.82 8.96 2.00 5.52 7.39 2.00 3.87 5.69QTIP + GuidedQuant (Ours) 2.00 6.11 7.99 2.00 5.33 7.05 2.00 3.80 5.61
GPTVQ 2D 3.13 5.63 7.32 3.13 4.87 6.45 3.13 3.38 5.18QuIP# 3.00 5.60 7.34 3.00 4.90 6.50 3.00 3.41 5.20AQLM 3.04 5.46 7.08 3.03 4.82 6.37 3.01 3.36 5.17QTIP 3.00 5.38 6.99 3.00 4.74 6.28 3.00 3.27 5.09QTIP + GuidedQuant (Ours) 3.00 5.28 6.87 3.00 4.71 6.22 3.00 3.25 5.08
GPTVQ 2D 4.13 5.24 6.77 4.13 4.65 6.13 4.13 3.18 5.01QuIP# 4.00 5.22 6.79 4.00 4.65 6.15 4.00 3.18 5.02AQLM 4.04 5.21 6.75 3.94 4.65 6.14 4.14 3.19 5.03QTIP 4.00 5.17 6.71 4.00 4.62 6.10 4.00 3.16 5.00QTIP + GuidedQuant (Ours) 4.00 5.16 6.68 4.00 4.61 6.09 4.00 3.15 5.00
Section
D.10
for results on different variants. We compareour approach with the following baselines: GPTVQ (
vanBaalen et al.
,
2024
), QuIP# (
Tseng et al.
,
2024a
), AQLM(
Egiazarian et al.
,
2024
), and QTIP (
Tseng et al.
,
2024b
).For QuIP#, AQLM, and QTIP, we report the results fromtheir respective papers, as they used the same or largercalibration datasets than ours. For GPTVQ, we report thereproduced results using our calibration data. Our methodconsistently outperforms all vector quantization baselinesacross different bit-widths and model sizes as well.5.2. Weight-and-activation QuantizationWeight-and-activation quantization methods apply uniformquantization on both weights and activations to leverage thefaster matrix multiplication units in the hardware (
Ashkbooset al.
,
2024
;
Liu et al.
,
2024
). State-of-the-art methods forweight-and-activation quantization include QuaRot (
Ashk-boos et al.
,
2024
) and SpinQuant (
Liu et al.
,
2024
), whichuse rotation matrices to reduce the activation outliers be-fore applying the uniform quantization. We incorporate ourGuidedQuant objective into the weight quantization processof these methods, guiding the model to quantize the weightsmore accurately. Specifically, we implement GuidedQuanton top of the SpinQuant using GPTQ weight quantizer andpresent the results in Table
5
. Following prior work, we usethe WikiText2 dataset (
Merity et al.
,
2016
) for calibration,with 128 sentences, each containing 2048 tokens (
Ashkbooset al.
,
2024
;
Liu et al.
,
2024
). Our objective consistentlyimproves the perplexity compared to the baseline methods,
Table 5.
Weight-and-activation quantization results on Llama-2models. L-2-7B, L-2-13B and L-2-70B denote Llama-2-7B,Llama-2-13B, and Llama-2-70B model, respectively. Wiki2 de-notes perplexity on Wikitext2 with the context size of 2048.
L-2-7B L-2-13B L-2-70B
Bits Method Wiki2# Wiki2# Wiki2#
16 Original 5.47 4.88 3.32
W4A4KV4QuaRot 6.08 5.39 3.80SpinQuant 5.95 5.24 3.71SpinQuant + GQuant (Ours) 5.89 5.19 3.71
W4A4KV16QuaRot 6.02 5.34 3.77SpinQuant 5.90 5.22 3.68SpinQuant + GQuant (Ours) 5.84 5.17 3.68
demonstrating its effectiveness.6. Additional Related WorkThere's a large body of work on neural network compression,even when considering only quantization for LLMs, makinga complete overview infeasible. Instead, we focus here onthe works most related to ours.Hessian-based Compression Neural networks compres-sion based on the second-order Taylor approximation of theend loss (Eq(
2
)) dates back to the early works of
LeCunet al.
(
1989
) and
Hassibi & Stork
(
1992
). OBD (
LeCunet al.
,
1989
) introduced this approach for pruning, underthe assumption that the Hessian matrix is diagonal. OBS
9
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
(
Hassibi & Stork
,
1992
) improved upon this by droppingthe diagonal assumption and instead approximating the Hes-sian by the empirical Fisher information matrix. However,applying OBS to large neural networks remains computa-tionally intractable. To address this, various more efficientHessian approximations have been proposed, including theK-FAC approximation (
Martens & Grosse
,
2015
;
Zeng &Urtasun
,
2018
;
Wang et al.
,
2019
;
Tycho F. A. van der Oud-eraa
,
2024
), block-diagonal Fisher approximation (
Singh &Alistarh
,
2020
;
Kurtic et al.
,
2022
;
Li et al.
,
2021
), and diag-onal Fisher approximation (
Choi et al.
,
2017
;
Theis et al.
,
2018
;
Kim et al.
,
2024
;
Bai et al.
,
2024
). Other strategiesdirectly estimate inverse-Hessian vector products (
Frantaret al.
,
2021
). The most similar approaches to GuidedQuantare ones that employ block-diagonal Fisher approximation,which achieve a good trade-off between approximation ac-curacy and computation and storage cost. However, thesemethods remain intractable at the scale of modern LLMs(see Section
D.11
).Gradient-based Compression Various compressionmethods are based on a first-order Taylor approximationof the end loss, with respect to output feature maps or gatesapplied to them (
Molchanov et al.
,
2017
;
2019
;
You et al.
,
2019
), or weights (
Ding et al.
,
2019
). The one most similarto GuidedQuant is (
Molchanov et al.
,
2019
), which employsthe same criterion in Eq.(
4
)to prune filters and neurons invision models. However, as explained in Section
3.2
, adopt-ing this criterion for quantizing modern LLMs is infeasible,without the averaging approximation we propose.Non-uniform Scalar PTQ for LLMs PTQ encompassesa vast array of work, so we focus on non-uniform scalarPTQ methods for LLMs that use look-up tables (codebooks)for weight decoding, which are closely related to our LNQalgorithm. One approach is zero-shot quantization, whichrequires no calibration data: Dynamic Tree Quantization(
Dettmers et al.
,
2021
) defines a new data type with dynamicexponential bits and stores decoded values in the codebook;Quantile Quantization (
Dettmers & Zettlemoyer
,
2023
)saves quantile values of the weight distribution; and QLoRA(
Dettmers et al.
,
2023
) introduces the NF4 data type usingquantiles of a standard normal distribution. These methodsshare a global codebook, with each layer maintaining itsown scale parameters. HIGGS (
Malinovskii et al.
,
2024b
)further refines this by adopting MSE-optimal grids for thestandard normal distribution and applying rotation matricesto approximate Gaussian weight distributions. Another lineof work involves one-shot quantization methods that opti-mize the output quantization error using calibration data.For instance, SqueezeLLM (
Kim et al.
,
2024
) optimizes sep-arate channel-wise codebooks via thek-means algorithm,while a 1D variant of GPTVQ (
van Baalen et al.
,
2024
)alternates between optimizing assignments with the GPTQ
algorithm and refining codebooks with gradient descent.The GPTVQ 1D shows the strongest performance amongthis line of research. Although not a scalar PTQ method,the vector quantization variant of AQLM (
Egiazarian et al.
,
2024
) also follows a similar paradigm, optimizing assign-ments through CD and codebooks via gradient descent.7. ConclusionWe introduced GuidedQuant, a novel PTQ approach thatintegrates gradient information from the end loss whilepreserving cross-weight dependencies within output chan-nels. GuidedQuant improves state-of-the-art methods acrossquantization formats, including weight-only scalar, weight-only vector, and weight-and-activation quantization. Fur-thermore, we identified inefficiencies in the current state-of-the-art methods for non-uniform scalar quantization andproposed LNQ, a new algorithm that, when combined withGuidedQuant, improves over the state-of-the-art perfor-mance. These contributions advance the efficiency andaccuracy of quantization for modern LLMs.Impact StatementThis work advances the compression of LLMs, in particularvia post-training quantization. As discussed, quantization,and model compression more broadly, reduces the memoryand computational requirements of LLMs and speeds upinference, thus reducing their environmental impact andenabling their use on resource-constrained devices and forlatency-critical applications. This can also help democratizeaccess to these models for organizations with limited re-sources and support privacy-preserving, offline applications.On the other hand, compression methods, including quan-tization, can adversely affect fairness in language models(
Ramesh et al.
,
2023
). While there are ongoing efforts to ad-dress fairness concerns in pruned LLMs (
Zayed et al.
,
2024
),extending these mitigation strategies to quantized modelsremains an important direction for future research. Fur-thermore, reducing the cost of using LLMs can also lowerthe barrier to their use by malicious actors. Finally, theenergy and resources saved through compression might bereinvested elsewhere, so the net reduction in environmentalharm is not guaranteed (Jevons paradox (
Alcott
,
2005
)).AcknowledgementsThis work was supported by Samsung Electronics Co., Ltd.(IO250418-12669-01), Mobile eXperience (MX) Business,Samsung Electronics Co., Ltd., Institute of Information &Communications Technology Planning & Evaluation (IITP)grant funded by the Korea government (MSIT) [No. RS-2020-II200882, (SW STAR LAB) Development of deploy-able learning intelligence via self-sustainable and trustwor-
10
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
thy machine learning, No. RS-2021-II211343, ArtificialIntelligence Graduate School Program (Seoul National Uni-versity), and No. 2022-0-00480, RS-2022-II220480, De-velopment of Training and Inference Methods for Goal-Oriented Artificial Intelligence Agents], and Basic ScienceResearch Program through the National Research Founda-tion of Korea (NRF) funded by the Ministry of Education(RS-2023-00274280). Hyun Oh Song is the correspondingauthor.References
Alcott, B. Jevons' paradox. Ecological economics, 54(1):921, 2005.
Ansel, J., Yang, E., He, H., Gimelshein, N., Jain, A., Voz-nesensky, M., Bao, B., Bell, P., Berard, D., Burovski, E.,et al. Pytorch 2: Faster machine learning through dynamicpython bytecode transformation and graph compilation.In ASPLOS, pp. 929947, 2024.
Arora, S., Babai, L., Stern, J., and Sweedyk, Z. The hardnessof approximate optima in lattices, codes, and systemsof linear equations. Journal of Computer and SystemSciences, 54(2):317331, 1997.
Arthur, D. and Vassilvitskii, S. k-means++: the advantagesof careful seeding. In SODA, 2007.
Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B.,Cameron, P., Jaggi, M., Alistarh, D., Hoefler, T., andHensman, J. Quarot: Outlier-free 4-bit inference in ro-tated llms. In NeurIPS, 2024.
Bai, R., Liu, Q., and Liu, B. Skim: Any-bit quantizationpushing the limits of post-training quantization. arXivpreprint arXiv:2412.04180, 2024.
Behdin, K., Acharya, A., Gupta, A., Song, Q., Zhu, S.,Keerthi, S., and Mazumder, R. Quantease: Optimization-based quantization for language models. arXiv preprintarXiv: 2309.01885, 2023.
Bisk, Y., Zellers, R., Bras, R. L., Gao, J., and Choi, Y.Piqa: Reasoning about physical commonsense in naturallanguage. In AAAI, 2020.
Chee, J., Cai, Y., Kuleshov, V., and De Sa, C. M. Quip: 2-bitquantization of large language models with guarantees.In NeurIPS, 2024.
Choi, Y., El-Khamy, M., and Lee, J. Towards the limit ofnetwork quantization. In ICLR, 2017.
Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins,M., and Toutanova, K. Boolq: Exploring the surprisingdifficulty of natural yes/no questions. In NAACL, 2019.
Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A.,Schoenick, C., and Tafjord, O. Think you have solvedquestion answering? try arc, the ai2 reasoning challenge.arXiv preprint arXiv:1803.05457, 2018.
Computer, T. Redpajama: an open dataset for traininglarge language models, 2023. URL
https://github. com/togethercomputer/RedPajama-Data
.
Dettmers, T. and Zettlemoyer, L. The case for 4-bit preci-sion: k-bit inference scaling laws. In ICML, 2023.
Dettmers, T., Lewis, M., Shleifer, S., and Zettlemoyer, L.8-bit optimizers via block-wise quantization. In ICLR,2021.
Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer,L. Qlora: Efficient finetuning of quantized llms. InNeurIPS, 2023.
Ding, X., Zhou, X., Guo, Y., Han, J., Liu, J., et al. Globalsparse momentum sgd for pruning very deep neural net-works. In NeurIPS, 2019.
Egiazarian, V., Panferov, A., Kuznedelev, D., Frantar, E.,Babenko, A., and Alistarh, D. Extreme compression oflarge language models via additive quantization. In ICML,2024.
Frantar, E. and Alistarh, D. Optimal brain compression:A framework for accurate post-training quantization andpruning. In NeurIPS, 2022.
Frantar, E., Kurtic, E., and Alistarh, D. M-fac: Efficientmatrix-free approximations of second-order information.In NeurIPS, 2021.
Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. Optq:Accurate post-training quantization for generative pre-trained transformers. In ICLR, 2023.
Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi,A., Foster, C., Golding, L., Hsu, J., Le Noac'h, A., Li,H., McDonell, K., Muennighoff, N., Ociepa, C., Phang,J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika,L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou,A. A framework for few-shot language model evaluation,07 2024. URL
https://zenodo.org/records/12608602
.
Gholami, A., Yao, Z., Kim, S., Hooper, C., Mahoney, M. W.,and Keutzer, K. Ai and memory wall. IEEE Micro, 2024.
Gray, A. Getting started with cuda graphs, 2019.URL
https://developer.nvidia.com/blog/cuda-graphs/
.
11
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Grønlund, A., Larsen, K. G., Mathiasen, A., Nielsen, J. S.,Schneider, S., and Song, M. Fast exact k-means, k-medians and bregman divergence clustering in 1d. arXivpreprint arXiv:1701.07204, 2017.
Hassibi, B. and Stork, D. Second order derivatives fornetwork pruning: Optimal brain surgeon. In NeurIPS,1992.
He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learn-ing for image recognition. arxiv e-prints. arXiv preprintarXiv:1512.03385, 10:9, 2015.
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M.,Song, D., and Steinhardt, J. Measuring massive multitasklanguage understanding. In ICLR, 2021.
Hyun, J. Log-time k-means clustering for 1d data: Novel ap-proaches with proof and implementation. arXiv preprintarXiv:2412.15295, 2024.
Kim, S., Hooper, C., Gholami, A., Dong, Z., Li, X., Shen,S., Mahoney, M., and Keutzer, K. Squeezellm: Dense-and-sparse quantization. In ICML, 2024.
Kurtic, E., Campos, D., Nguyen, T., Frantar, E., Kurtz, M.,Fineran, B., Goin, M., and Alistarh, D. The optimal bertsurgeon: Scalable and accurate second-order pruning forlarge language models. In EMNLP, 2022.
LeCun, Y., Denker, J., and Solla, S. Optimal brain damage.In NeurIPS, 1989.
Li, Y., Gong, R., Tan, X., Yang, Y., Hu, P., Zhang, Q., Yu,F., Wang, W., and Gu, S. Brecq: Pushing the limit ofpost-training quantization by block reconstruction. InICLR, 2021.
Liu, Z., Zhao, C., Fedorov, I., Soran, B., Choudhary, D., Kr-ishnamoorthi, R., Chandra, V., Tian, Y., and Blankevoort,T. Spinquant: Llm quantization with learned rotations.arXiv preprint arXiv:2405.16406, 2024.
Lloyd, S. Least squares quantization in pcm. IEEE transac-tions on information theory, 28(2):129137, 1982.
Malinovskii, V., Mazur, D., Ilin, I., Kuznedelev, D.,Burlachenko, K., Yi, K., Alistarh, D., and Richtarik, P. Pv-tuning: Beyond straight-through estimation for extremellm compression. In NeurIPS, 2024a.
Malinovskii, V., Panferov, A., Ilin, I., Guo, H., Richt´arik,P., and Alistarh, D. Pushing the limits of large lan-guage model quantization via the linearity theorem. arXivpreprint arXiv:2411.17525, 2024b.
Martens, J. and Grosse, R. Optimizing neural networkswith kronecker-factored approximate curvature. In ICML,2015.
Merity, S., Xiong, C., Bradbury, J., and Socher, R.Pointer sentinel mixture models. arXiv preprintarXiv:1609.07843, 2016.
Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can asuit of armor conduct electricity? a new dataset for openbook question answering. In EMNLP, 2018.
Molchanov, P., Tyree, S., Karras, T., Aila, T., and Kautz,J. Pruning convolutional neural networks for resourceefficient inference. In ICLR, 2017.
Molchanov, P., Mallya, A., Tyree, S., Frosio, I., and Kautz,J. Importance estimation for neural network pruning. InProceedings of the IEEE/CVF conference on computervision and pattern recognition, pp. 1126411272, 2019.
Nagel, M., Amjad, R. A., Van Baalen, M., Louizos, C., andBlankevoort, T. Up or down? adaptive rounding for post-training quantization. In ICML, pp. 71977206. PMLR,2020.
Nair, P. A. and Suggala, A. S. Cdquant: Greedy coordinatedescent for accurate llm quantization. arXiv preprintarXiv: 2406.17542, 2024.
Park, G., Park, B., Kim, M., Lee, S., Kim, J., Kwon, B.,Kwon, S. J., Kim, B., Lee, Y., and Lee, D. Lut-gemm:Quantized matrix multiplication based on luts for efficientinference in large-scale generative language models. InICML, 2024a.
Park, Y., Hyun, J., Cho, S., Sim, B., and Lee, J. W.Any-precision llm: Low-cost deployment of multiple,different-sized llms. In ICML, 2024b.
Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S.,Matena, M., Zhou, Y., Li, W., and Liu, P. Exploringthe limits of transfer learning with a unified text-to-texttransformer. Journal of Machine Learning Research, 21(140):167, 2020.
Ramesh, K., Chavan, A., Pandit, S., and Sitaram, S. Acomparative study on the impact of model compressiontechniques on fairness in language models. In ACL, pp.1576215782, 2023.
Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y.Winogrande: An adversarial winograd schema challengeat scale. arXiv preprint arXiv: 1907.10641, 2019.
Sap, M., Rashkin, H., Chen, D., LeBras, R., and Choi, Y.Socialiqa: Commonsense reasoning about social interac-tions. In EMNLP, 2019.
Singh, S. P. and Alistarh, D. Woodfisher: Efficient second-order approximation for neural network compression. InNeurIPS, 2020.
12
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Theis, L., Korshunova, I., Tejani, A., and Husz´ar, F. Fastergaze prediction with dense networks and fisher pruning.arXiv preprint arXiv:1801.05787, 2018.
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E.,Azhar, F., et al. Llama: Open and efficient foundation lan-guage models. arXiv preprint arXiv:2302.13971, 2023.
Tseng, A., Chee, J., Sun, Q., Kuleshov, V., and De Sa,C. Quip#: Even better llm quantization with hadamardincoherence and lattice codebooks. In ICML, 2024a.
Tseng, A., Sun, Q., Hou, D., and De Sa, C. Qtip: Quan-tization with trellises and incoherence processing. InNeurIPS, 2024b.
Tycho F. A. van der Ouderaa, Markus Nagel, M. V. B. T. B.The llm surgeon. In ICLR, 2024.
van Baalen, M., Kuzmin, A., Nagel, M., Couperus, P., Bas-toul, C., Mahurin, E., Blankevoort, T., and Whatmough,P. Gptvq: The blessing of dimensionality for llm quanti-zation. arXiv preprint arXiv:2402.15319, 2024.
Wang, C., Grosse, R., Fidler, S., and Zhang, G. Eigen-damage: Structured pruning in the kronecker-factoredeigenbasis. In ICML, 2019.
You, Z., Yan, K., Ye, J., Ma, M., and Wang, P. Gate deco-rator: Global filter pruning method for accelerating deepconvolutional neural networks. In NeurIPS, 2019.
Zayed, A., Mordido, G., Shabanian, S., Baldini, I., andChandar, S. Fairness-aware structured pruning in trans-formers. In AAAI, 2024.
Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi,Y. Hellaswag: Can a machine really finish your sentence?In ACL, 2019.
Zeng, W. and Urtasun, R. Mlprune: Multi-layer pruningfor automated neural network compression. OpenReview,2018.
13
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
A. ProofsHere, we prove Remark
3.1
and Proposition
4.1
, each restated here for convenience.
Remark 3.1.
The sum of the layer-wise objective in Eq.(
4
)over all layers is equal to the following quadratic approximationof the change in the end lossnLXl=1d(l)outXj=1(w(l)jbw(l)j)>F(l)j(w(l)jbw(l)j): (5)
Proof.
Recall thatZ(l)= X(l)W(l). Then, by chain rule we have that@`i
@w(l)j=@`i
@Z(l)ij(X(l)i;:)>. Note also that@`
@Z(l)ij=@`i
@Z(l)ij.Hence,@`
@Z(l) (X(l)W(l) X(l)cW(l))2F=d(l)outXj=1nXi=1 @`i
@Z(l)ijX(l)i;:(w(l)jbw(l)j)!2 =d(l)outXj=1nXi=10@ @`i
@w(l)j!>(w(l)jbw(l)j)1A2 = nd(l)outXj=1(w(l)jbw(l)j)>F(l)j(w(l)jbw(l)j);where the last equality follows from the definition of the Fisher blocksF(l)j=1
nPni=1(@`i
@w(l)j)(@`i
@w(l)j)>. Taking the sumover l 2 [L] on both sides yields the claim.
Proposition 4.1.
For anyj 2 [dout], letfj(c; P) = kXwj XPck22, and letc(j)tandP(j)tdenotec(j)andP(j)at thet-thiteration of LNQ. Then,fj(c(j)t; P(j)t) fj(c(j)t+1; P(j)t) fj(c(j)t+1; P(j)t+1)for allt, and the sequenceffj(c(j)t; P(j)t)gt1 converges.
Proof.
We first show that the objective value is non-increasing in LNQ. For allt 1, we haveP(j)t1m= 1dinand thusthe corresponding quantized weights^wj= P(j)tc(j)tare feasible Hence, CD is initialized with a feasible solution at eachiteration t, so it acts as a descent method. Then,fj(c(j)t; P(j)t) fj(c(j)t+1; P(j)t) (since c(j)t+1= argminc(j)2Rmfj(c; Pt)) fj(c(j)t+1; P(j)t+1); (since CD does not increase the objective value)for allt 1. Sincefj(c; P)is bounded below by0, the sequenceffj(c(j)t; P(j)t)gis monotonically non-increasing andbounded below. Hence, it converges to its infimum by the monotone convergence theorem.
B. Hyperparameters and DetailsIn this section, we clarify the hyperparameters and details of the methods discussed in the main paper.B.1. GuidedQuantThe proposed GuidedQuant method has a single hyperparameter: the number of groupgused to average the Hessianmatrices
Hj(see Section
3.2
). For weight-only quantization experiments, we setg = 4for Llama-2-7B and Llama-2-13B,andg = 2for Llama-2-70B. For weight-and-activation quantization experiments, we setg = 1. For the hyperparameterg,we selected the number of groups to be as large as possible within the limits of our computational and memory constraints.Notably, GuidedQuant also maintains strong performance with smaller values of g (see Section
D.5
).Computing the Hessian (Line 4 in Algorithm
1
) and running the quantization algorithmQ(Line 5 in Algorithm
1
) for eachgroup and layer can be parallelized. We parallelize Hessian computation across groups. For quantization, we parallelize
14
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Algorithm 3 Efficient CD algorithm with precomputation
input
Hessian of the objectiveH 2 Rdinfidin, input weightW 2 Rdinfidout, current codebookc(j)2 Rmand currentquantized weightcW 2 Rdinfidout. Initialize Q 2 f1; : : : ; mgdinfidout(rounded indices).
1:
eH diag(H)1H , U StrictUpper(eH).
2:
for k = 1 to K do
3:
B U(cW W)
4:
for i = 1 to dindo
5:
cWi;: Round(Wi;: Bi;:), Qi;: RoundIdx(Wi;: Bi;:)
6:
B(i+1):;: B(i+1):;:+ U(i+1):;i(cWi;: Wi;:)
7:
end for
8:
end for
9:
8i 2 [din]; j 2 [dout]; q 2 [m] : P(j)iq=(1 if q = Qij;0 otherwise.fExtracting assignment matrixg
output
P(1); : : : ; P(dout).
across groups in LNQ + GuidedQuant, while in QTIP + GuidedQuant and SpinQuant + GuidedQuant, we run this step in asequential manner to minimally change the codebase of the original methods.B.2. LNQThe proposed LNQ method has two hyperparameters: (1) the number of iterations during which we alternate betweenoptimizingcandP(Tin Algorithm
2
), and (2) the number of coordinate descent iterations over the output dimensions (Kin Algorithm
2
). For Llama-2-7B and Llama-2-13B, we useT = 2andK = 4, and for Llama-2-70B, we useT = 1andK = 4 in all the experiments.We further explain a derivation of the time complexity of the proposed LNQ algorithm (Algorithm
2
), discussed in Section
4.2
.First, the time complexity of the Cholesky decomposition for a matrix H 2 Rdinfidinis O(d3in) (Line 1). For optimizing the codebook (Line 4 and 14), we analyze the computational cost within the loop as follows:

Computing L>P(j)requires O(d2inm) time.

Computing L>wjrequires O(d2in) time.

torch.linalg.lstsqfunction uses QR decomposition ofL>P(j)to compute least squares solution, whichrequires O(dinm2) time. Since din m, the dominant cost is O(d2inm). For computing^wj= P(j)c(j)(Line 5), the cost is O(dinm).In CD, the cost of the minimizing the objective for each coordinatei(Line 8, Eq.(
11
)) isO(din+ m). Sincedin m, thedominant cost is O(din). Considering loop iterations, optimizing the code (Lines 6-12) takes O(d2inK) time complexity.Therefore, the cost of Lines 4-12 isO(d2in(m + K)), and the cost of Lines 2-15 isO(d2indoutT (m + K)). Including theCholesky decomposition, the total time complexity of LNQ algorithm is O(d3in+ d2indoutT (m + K)). B.3. Efficient Implementation of CD Algorithm in LNQIn the LNQ algorithm (Algorithm
2
), computing the solution across all output channelsj 2 [dout]is independent and thusfully parallelizable. Therefore, we perform the coordinate descent (CD) updates for each output channel in parallel.Coordinate-wise Closed-form Solution. For a given quantized weight matrixcW 2 Rdinfidout, the CD update for thei-thinput coordinate can be computed in parallel using the coordinate-wise closed-form solution as follows (
Behdin et al.
,
2023
,
15
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
This acceleration trick has been proposed in QuIP (
Chee et al.
,
2024
, Appendix B.2.) and QuantEase (
Behdin et al.
,
2023
).It is worth noting that this precomputation trick does not change the theoretical time complexity, but improves practicalperformance by exploiting the GPU parallelization. In particular, the CD update for thei-th coordinate in Equation (
12
)requires2dout(din 1)FLOPs without precomputation, while with precomputation, the cost is reduced to2dout(din i) FLOPs.Lazy Batch-updates. After incorporating the precomputation trick, we observe that the update steps within the CD loop(Lines 47 in Algorithm
3
) resemble the OBQ update scheme used in the GPTQ method (
Frantar & Alistarh
,
2022
;
Frantaret al.
,
2023
). In OBQ, each iteration involves rounding a single coordinate and adjusts the not-yet-rounded coordinatesaccordingly. Analogously, our CD update with precomputation roundsWi;: Bi;:for thei-th coordinate and incrementallyupdates B(i+1):;:to reflect the new values ofcWi;:.Both OBQ and our CD update suffer from a low compute-to-memory ratio: although each iteration involves relatively fewFLOPs, it requires frequent reading and writing to large matrices. As a result, these updates tend to be memory-bound andsuffer from poor GPU utilization. To mitigate this, GPTQ introduces lazy batch-updates, in which a batch of coordinates(with batch sizeb = 128) is processed together. Within each batch, updates are applied sequentially to each coordinate,while corrections are made only for the remaining unprocessed coordinates within the batch. Once allbcoordinates in thebatch are updated, a global correction step is performed for the rest of the matrix. This strategy improves memory efficiencyby reducing the frequency of global updates.We adopt this lazy batch-updates approach in our CD implementation with precomputation trick. Specifically, we restrictupdates to the relevant portion ofBwithin each block ofbcoordinates, and defer global updates toBuntil the entire blockhas been processed. This significantly reduces memory-bound operations and enhances GPU utilization. The final efficientCD algorithm incorporating both precomputation trick and lazy batch-updates is given in Algorithm
4
.QuIP (
Chee et al.
,
2024
) also supports lazy batch-updates in their open-source code, though it is not mentioned in theirpaper. QuantEase (
Behdin et al.
,
2023
) does not use this approach in their implementation. As with the precomputationtrick, lazy batch-updates do not change the theoretical time complexity. However, they substantially accelerate the overallalgorithm in practice by better utilizing GPU resources.Speedup Factor To demonstrate the speedup achieved by our optimization techniques for the CD algorithm, we reportthe quantization time for quantizing the Llama-2-7B model into 4-bit precision on a single RTX 6000 Ada GPU. Withoutany optimizations, adopting the naive strategy of exhaustively evaluating the objective function for all coordinate choicesand selecting the option with the lowest value takes 3.9 hours to quantize the entire model. Applying the coordinate-wiseclosed-form solution described in Eq.(
12
)reduces this time to 2.7 hours. Incorporating the precomputation trick furtherlowers it to 1.2 hours. Finally, applying lazy batch-updates brings the total quantization time down to just 0.9 hours. Overall,these optimizations yield more than a 4fi speedup in end-to-end quantization time on GPU.B.4. GPTVQIn the original GPTVQ paper (
van Baalen et al.
,
2024
), the authors used 128 sentences from the WikiText2 dataset (
Merityet al.
,
2016
), each containing 2048 tokens, as a calibration data. For a fair comparison, we reproduced their method usingtheir open-sourced code but used 1024 sentences of RedPajama dataset (
Computer
,
2023
), each containing 4096 tokens. Weadopted their default hyperparameters except for the group size and block size, which we adjusted to match the average bitwidth when comparing with different methods in Table
3
. We provide a complete list of GPTVQ hyperparameters for eachtable in Table
6
.C. Details on Experimental SetupThis section provides a detailed explanation of the experimental settings used.C.1. End-to-end Inference Throughput Experiments (Table
2
)In Table
2
, we measure each model's inference throughput in generating 100 tokens on RTX 4090 GPU, after integrating thekernels with into a PyTorch-based inference pipeline optimized with thetorch.compilefunction (
Ansel et al.
,
2024
;
Gray
,
2019
). For QTIP, our chosen vector quantization kernel, we adopt the HYB variant of it as its GPU kernel is publicly
17
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Table 6. Hyperparameters that we used in reproducing GPTVQ (
van Baalen et al.
,
2024
) in Table
3
and Table
4
.
Weight VQ Codebook sharing Scaling CodebookTable bits dim group size block size bit-width Avg bits
Table
3
2 1 1024  8 2.033 1 2048  8 3.034 1 8192 256 8 4.034 1 4096 128 8 4.06
Table
4
2 2 2048  8 2.132 4 32768  8 2.253 2 16384 64 8 3.134 2 65536 64 8 4.13
Table 7.
End-to-end inference throughput of Llama-2 models on RTX 4090 GPU, including the vector quantization kernel after fusing thequery/key/value projection matrices into one linear layer and the up/gate projection matrices into another when measuring the throughput.OOM indicates an Out-of-Memory error, meaning the GPU lacks memory to run model inference.
Llama-2-7B Llama-2-13B Llama-2-70B
Type Bits# Tok/s" Bits# Tok/s" Bits# Tok/s"
Original 16 67 16 OOM 16 OOM
Uniform scalar 2.00 334 2.00 200 2.00 47Non-uniform scalar 2.01 347 2.01 203 2.01 47Vector 2.00 200 2.00 121 2.00 38Vector (fused) 2.00 248 2.00 153 2.00 42
Uniform scalar 3.00 260 3.00 150 3.00 OOMNon-uniform scalar 3.03 264 3.02 148 3.01 OOMVector 3.00 176 3.00 103 3.00 OOMVector (fused) 3.00 209 3.00 123 3.00 OOM
Uniform scalar 4.00 214 4.00 121 4.00 OOMNon-uniform scalar 4.05 209 4.04 116 4.03 OOMVector 4.00 151 4.00 89 4.00 OOMVector (fused) 4.00 176 4.00 103 4.00 OOM
available, though it is possible to implement fast GPU kernels with other variants as well (
Tseng et al.
,
2024b
).For the base model and for models quantized using uniform or non-uniform scalar formats, we fuse the query/key/valueprojection matrices into one linear layer and the up/gate projection matrices into another when measuring the throughput.This fusion trick can be applied to QTIP as well, provided the matrices are fused before quantization and the scale parametersare shared across layers. However, in the main paper, we present QTIP results without fusion to match the originalexperimental setup (and reported numbers) from their work, in which they quantize the layers independently without fusingthem. Meanwhile, scalar quantization methods quantize the layer in an output channel-wise manner, and this allows fusingmatrices even when layers are quantized separately.For completeness, we include Table
7
, which also shows QTIP's fused end-to-end throughput (measured using dummyvalues) to illustrate the impact of fusion, restating the relevant results from Table
2
. Although the fusion boosts thethroughput, it does not change the conclusion that QTIP still runs more slowly than the scalar quantization methods.C.2. Implementation Details for Different Quantization TypesGuidedQuant employs a quantization algorithmQas a subroutine (Line 8 in Algorithm
1
). In this section, we clarifywhich specific quantization algorithmQeach method uses, which GuidedQuant builds upon in Algorithm
1
. We integrate
18
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Table 8.
Total GPU cost incurred during the quantization process for LNQ and QTIP, both with and without GuidedQuant, across variousgroup sizes g. We specify the number and type of GPU used in the parentheses. R6A denotes the RTX 6000 Ada GPU.
LNQ QTIP
Model Method GPU Cost - 2 bits GPU Cost - 3 bits GPU Cost - 4 bits GPU Cost - 2 bits GPU Cost - 3 bits GPU Cost - 4 bits
Llama-2-7B Layer-wise (LNQ, QTIP) 0.5 h (1fiR6A) 0.6 h (1fiR6A) 0.9 h (1fiR6A) 1.3 h (1fiR6A) 1.2 h (1fiR6A) 1.2 h (1fiR6A)Layer-wise + GQuant (g = 1) 0.5 h (1fiR6A) 0.6 h (1fiR6A) 0.9 h (1fiR6A) 1.3 h (1fiR6A) 1.2 h (1fiR6A) 1.2 h (1fiR6A)Layer-wise + GQuant (g = 2) 0.6 h (1fiR6A) 0.7 h (1fiR6A) 0.9 h (1fiR6A) 1.5 h (1fiR6A) 1.4 h (1fiR6A) 1.5 h (1fiR6A)Layer-wise + GQuant (g = 4) 0.7 h (1fiR6A) 0.7 h (1fiR6A) 0.9 h (1fiR6A) 1.9 h (1fiR6A) 1.9 h (1fiR6A) 1.9 h (1fiR6A)
Llama-2-13B Layer-wise (LNQ, QTIP) 0.9 h (1fiR6A) 1.1 h (1fiR6A) 1.6 h (1fiR6A) 3.0 h (1fiR6A) 2.7 h (1fiR6A) 2.7 h (1fiR6A)Layer-wise + GQuant (g = 1) 0.9 h (1fiR6A) 1.1 h (1fiR6A) 1.6 h (1fiR6A) 3.0 h (1fiR6A) 2.7 h (1fiR6A) 2.7 h (1fiR6A)Layer-wise + GQuant (g = 2) 1.1 h (1fiR6A) 1.2 h (1fiR6A) 1.6 h (1fiR6A) 2.4 h (1fiR6A) 2.2 h (1fiR6A) 2.3 h (1fiR6A)Layer-wise + GQuant (g = 4) 1.2 h (1fiR6A) 1.3 h (1fiR6A) 1.7 h (1fiR6A) 3.0 h (1fiR6A) 3.0 h (1fiR6A) 3.0 h (1fiR6A)
Llama-2-70B Layer-wise (LNQ, QTIP) 2.6 h (1fiR6A) 3.3 h (1fiR6A) 5.1 h (1fiR6A) 12.0 h (1fiR6A) 10.8 h (1fiR6A) 11.0 h (1fiR6A)Layer-wise + GQuant (g = 1) 2.6 h (1fiR6A) 3.3 h (1fiR6A) 5.1 h (1fiR6A) 12.0 h (1fiR6A) 10.8 h (1fiR6A) 11.0 h (1fiR6A)Layer-wise + GQuant (g = 2) 3.7 h (1fiR6A) 4.7 h (1fiR6A) 6.8 h (1fiR6A) 13.0 h (1fiR6A) 11.9 h (1fiR6A) 12.0 h (1fiR6A)
Table 9.
Total GPU cost and disk usage incurred during the gradient and Hessian caching processes for each objectiveweightedk-means(SqueezeLLM), layer-wise (LNQ, QTIP), and GuidedQuant. We specify the number and type of GPU used in the parentheses. R6A andA100 denote the RTX 6000 Ada GPU and the A100 GPU, respectively. The calibration data are 1024 sentences of the RedPajama dataset,each containing 4096 tokens.
Gradient Caching Hessian Caching
Model Method GPU Cost Disk Size GPU Cost Disk Size
Llama-2-7B Weighted k-means (SqueezeLLM) 0.3 h (1fiA100) 13 GiB  Layer-wise (LNQ, QTIP)   0.3 h (4fiR6A) 27 GiBLayer-wise + GQuant (g = 1) 0.3 h (1fiA100) 2 GiB 0.3 h (4fiR6A) 27 GiBLayer-wise + GQuant (g = 2) 0.3 h (1fiA100) 4 GiB 0.4 h (4fiR6A) 53 GiBLayer-wise + GQuant (g = 4) 0.3 h (1fiA100) 7 GiB 0.8 h (4fiR6A) 106 GiB
Llama-2-13B Weighted k-means (SqueezeLLM) 0.6 h (2fiA100) 25 GiB  Layer-wise (LNQ, QTIP)   0.5 h (4fiR6A) 52 GiBLayer-wise + GQuant (g = 1) 0.6 h (2fiA100) 3 GiB 0.5 h (4fiR6A) 52 GiBLayer-wise + GQuant (g = 2) 0.6 h (2fiA100) 7 GiB 0.9 h (4fiR6A) 104 GiBLayer-wise + GQuant (g = 4) 0.6 h (2fiA100) 13 GiB 1.5 h (4fiR6A) 208 GiB
Llama-2-70B Weighted k-means (SqueezeLLM) 2.7 h (6fiA100) 129 GiB  Layer-wise (LNQ, QTIP)   3.5 h (4fiR6A) 366 GiBLayer-wise + GQuant (g = 1) 2.7 h (6fiA100) 5 GiB 3.5 h (4fiR6A) 366 GiBLayer-wise + GQuant (g = 2) 2.7 h (6fiA100) 9 GiB 5.8 h (4fiR6A) 731 GiB
GuidedQuant with three different quantization methods: (1) LNQ for weight-only scalar quantization, (2) QTIP for weight-only vector quantization, and (3) SpinQuant for weight-and-activation quantization. LNQ adopts the algorithm shown inAlgorithm
2
, QTIP uses the BlockLDLQ algorithm proposed in
Tseng et al.
(
2024a
), and SpinQuant employs the GPTQalgorithm introduced in
Frantar et al.
(
2023
).D. Additional Results and DiscussionsD.1. Quantization CostIn this section, we present a detailed breakdown of the computational costs associated with our method, as summarized inTable
8
and Table
9
. The layer-wise quantization methods on which we build typically require two phases: (1) caching theHessian matrices to disk, and (2) loading them to quantize weights based on these cached Hessian matrices. It is worthnoting that the cost of the first phase (caching) can be amortized if one needs to quantize the same model multiple timesat different bit-widths or configurations, as the Hessian matrices can be reused. We report the weight quantization cost inTable
8
, and the Hessian-caching cost in Table
9
.From Table
8
, observe that forg = 1, the quantization is identical to standard layer-wise quantization, since the Hessian sizeis the same. Even forg = 2org = 4, the quantization cost does not increase by more than 50%. This is because whilemore Hessian matrices are employed, each weight block to be quantized becomes correspondingly smaller, leaving the total
19
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Table 10.
Weight-only scalar post-training quantization results on Llama-3 models. Wiki2 and C4 denotes perplexity on WikiText2 andC4, respectively. The perplexity is measured with the context size of 8192.
Llama-3-8B Llama-3-70B
Method Bits# Wiki2# C4# Bits# Wiki2# C4#
Original 16 5.54 7.10 16 2.59 5.78
SqueezeLLM 2.01 16322 1501 2.01 38.53 38.15LNQ (Ours) 2.01 133.00 72.75 2.01 24.22 19.71LNQ + GuidedQuant (Ours) 2.01 30.80 20.41 2.01 10.21 11.06
SqueezeLLM 3.03 7.39 8.84 3.02 4.12 6.44LNQ (Ours) 3.03 7.28 8.46 3.01 4.57 6.61LNQ + GuidedQuant (Ours) 3.03 6.99 8.10 3.01 3.90 6.27
SqueezeLLM 4.05 5.91 7.43 4.03 2.91 5.91LNQ (Ours) 4.05 5.90 7.40 4.03 3.05 5.94LNQ + GuidedQuant (Ours) 4.05 5.80 7.32 4.03 2.89 5.89
Table 11.
Weight-only scalar post-training quantization results on Llama-2 models, including end-to-end throughput. Wiki2 and C4denotes perplexity on WikiText2 and C4, respectively. The perplexity is measured with the context size of 4096. Throughput is evaluatedon an RTX 3090 GPU, reported as the average of 5 runs with standard deviation in parentheses. OOM indicates an Out-of-Memory error,meaning the GPU lacks memory to run model inference.
Llama-2-7B Llama-2-13B Llama-2-70B
Method Bits# Wiki2# C4# Tok/s" Bits# Wiki2# C4# Tok/s" Bits# Wiki2# C4# Tok/s"
Original 16 5.12 6.63 64.8 (0.1) 16 4.57 6.05 OOM 16 3.12 4.97 OOM
SqueezeLLM 2.01 39.58 44.05 245.1 (1.8) 2.01 16.24 19.20 140.5 (0.5) 2.01 9.17 13.03 31.5 (0.0)LNQ (Ours) 2.01 23.31 26.71 244.6 (0.6) 2.01 8.78 11.80 141.1 (0.4) 2.01 5.23 7.31 31.6 (0.1)LNQ + GuidedQuant (Ours) 2.01 8.83 11.15 244.4 (2.9) 2.01 7.26 9.17 141.2 (0.5) 2.01 5.04 7.04 31.6 (0.1)
SqueezeLLM 3.03 5.74 7.44 207.3 (1.6) 3.02 4.99 6.60 118.0 (0.5) 3.01 3.53 5.31 OOMLNQ (Ours) 3.03 5.89 7.74 207.3 (2.1) 3.02 5.02 6.68 118.0 (0.6) 3.01 3.50 5.31 OOMLNQ + GuidedQuant (Ours) 3.03 5.57 7.22 207.6 (1.7) 3.02 4.91 6.49 117.9 (0.6) 3.01 3.47 5.27 OOM
SqueezeLLM 4.05 5.23 6.78 161.8 (1.5) 4.04 4.67 6.15 89.8 (0.1) 4.03 3.20 5.04 OOMLNQ (Ours) 4.05 5.26 6.82 161.7 (1.6) 4.04 4.67 6.17 89.7 (0.1) 4.03 3.20 5.04 OOMLNQ + GuidedQuant (Ours) 4.05 5.21 6.75 162.0 (1.8) 4.04 4.65 6.14 89.8 (0.1) 4.03 3.20 5.03 OOM
computation unchanged. All these steps can be performed in an embarrassingly parallel manner; for example, quantizingLlama-2-70B using our LNQ algorithm takes less than three hours when using 8 RTX 6000 Ada GPUs.We further report the cost of caching Hessian matrices in Table
9
, along with the number of GPU used and the disk sizerequirements. Note that while we used 4 GPUs for caching, this process is also fully parallelizable; using fewer GPUs willsimply take longer (it can run on a single GPU), whereas additional GPUs can shorten the total time. Finally, our method'sdisk-space requirement is proportional to the number of groupsg. However, we highlight that for constrained disk space,choosing a smaller number of groups can still capture most of the performance benefits (Table
13
).D.2. Results on Llama-3 ModelsIn this section, we present the results of evaluating LNQ and LNQ combined with GuidedQuant on Llama-3-8B andLlama-3-70B models, comparing with SqueezeLLM under a weight-only scalar quantization setting. We present the resultsin Table
10
. We use RedPajama dataset (
Computer
,
2023
) for calibration with 1024 sentences, each containing 4096 tokens.We set the number of groups to beg = 1for Llama-3-8B and Llama-3-70B, and set the hyperparameters for LNQ (and LNQ+ GuidedQuant) to beT = 2; K = 4for Llama-3-8B andT = 1; K = 4for Llama-3-70B model. LNQ with GuidedQuantconsistently outperforms the baselines, demonstrating the robustness and effectiveness of our approach.
20
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Table 12.
Weight-only scalar post-training quantization results, evaluated on zero-shot and few-shot downstream tasks. Zero-shot Avgdenotes the average accuracy across eight zero-shot tasks: BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-easy, ARC-challenge,and OBQA. For the few-shot benchmark, MMLU (5-shot) denotes accuracy on the MMLU benchmark in a 5-shot setting. We report thestandard error in parentheses and bold the best results, as well as those whose accuracy score falls within the top scorestandard error.
Llama-2-7B Llama-2-13B
Method Bits# Zero-shot Avg" MMLU (5-shot)" Bits# Zero-shot Avg" MMLU (5-shot)"
Original 16 59.88 (0.43) 45.97 (0.41) 16 62.80 (0.43) 54.93 (0.40)
SqueezeLLM 2.01 41.80 (0.41) 24.75 (0.36) 2.01 42.44 (0.41) 24.47 (0.36)GPTVQ 1D 2.03 37.35 (0.40) 26.56 (0.37) 2.03 46.34 (0.41) 29.63 (0.38)LNQ (Ours) 2.01 40.30 (0.40) 26.76 (0.37) 2.01 49.51 (0.42) 32.51 (0.39)LNQ + GuidedQuant (Ours) 2.01 50.39 (0.43) 31.53 (0.39) 2.01 53.98 (0.43) 40.15 (0.41)
SqueezeLLM 3.03 57.55 (0.43) 40.59 (0.41) 3.02 61.16 (0.43) 49.94 (0.40)GPTVQ 1D 3.03 54.92 (0.43) 41.08 (0.41) 3.03 60.38 (0.43) 52.06 (0.40)LNQ (Ours) 3.03 56.85 (0.43) 42.18 (0.41) 3.02 60.61 (0.43) 51.62 (0.40)LNQ + GuidedQuant (Ours) 3.03 58.16 (0.43) 43.38 (0.41) 3.02 61.00 (0.43) 52.67 (0.40)
SqueezeLLM 4.05 59.41 (0.43) 44.79 (0.41) 4.04 62.32 (0.43) 54.52 (0.40)GPTVQ 1D 4.06 59.23 (0.43) 45.06 (0.41) 4.06 62.37 (0.43) 54.95 (0.40)LNQ (Ours) 4.05 59.14 (0.43) 44.51 (0.41) 4.04 62.40 (0.43) 54.79 (0.40)LNQ + GuidedQuant (Ours) 4.05 59.41 (0.43) 45.16 (0.41) 4.04 62.17 (0.43) 54.39 (0.40)
D.3. Additional Inference Throughput ResultsGuidedQuant leverages existing CUDA kernels (Any-Precision-LLM kernel (
Park et al.
,
2024b
) for weight-only scalar andQTIP kernel (
Tseng et al.
,
2024b
) for weight-only vector quantization) and optimizes assignment and codebook values, thusachieving improved performance without sacrificing inference throughput. To validate this, we compare weight-only scalarPTQ results on Llama-2 models across methods using the same CUDA kernel, as shown in Table
11
. Specifically, we reportperplexity and end-to-end throughput for SqueezeLLM, LNQ, and LNQ + GuidedQuant, all using the Any-Precision Kernel(
Park et al.
,
2024b
). Throughput is measured on an RTX 3090 GPU as the average of 5 runs, with standard deviation inparentheses. Results confirm that our methods (LNQ and LNQ + GuidedQuant) achieve better perplexity while maintainingthe same throughput as other method using the identical kernel.D.4. Evaluations on Zero-shot and Few-shot Downstream BenchmarksIn this section, we provide the evaluations on zero-shot and few-shot downstream tasks of our methods (LNQ and LNQ+ GuidedQuant) alongside baselines (SqueezeLLM and GPTVQ 1D) under the weight-only scalar quantization settings,using Llama-2-7B and Llama-2-13B models, in Table
12
. The evaluation includes eight zero-shot tasks: BoolQ (
Clarket al.
,
2019
), PIQA (
Bisk et al.
,
2020
), SIQA (
Sap et al.
,
2019
), HellaSwag (
Zellers et al.
,
2019
), WinoGrande (
Sakaguchiet al.
,
2019
), ARC-easy (
Clark et al.
,
2018
), ARC-challenge (
Clark et al.
,
2018
), and OBQA (
Mihaylov et al.
,
2018
). For afew-shot benchmark, we include results on the MMLU (
Hendrycks et al.
,
2021
) benchmark in a 5-shot setting. We evaluateon these tasks using version 0.4.3 of the lm-evaluation-harness library (
Gao et al.
,
2024
).Table
12
reports both accuracy and standard error for all methods. We highlight the best-performing results, as well asthose whose accuracy falls within the top scorestandard error, under the same bit width constraint. The results show thatLNQ combined with GuidedQuant consistently matches or surpasses baseline performance, with notable improvements inextreme quantization scenarios, such as 2-bit quantization.D.5. Results on Varying the Number of Groups gIn this section, we present results on how varying the number of groupsg(introduced in Section
3.2
) affects performance,focusing on whether fewer groups preserve accuracy or introduce trade-offs when averaging the Hessian within each group.Table
13
summarizes the impact of changinggunder a non-uniform scalar quantization scheme. While increasinggcanmoderately improve results in extreme cases (e.g., quantizing models into 2 bits), performance differences across the numberof groups remain minimal in other scenarios. Note that for weight-only quantization experiments, we choseg = 4for
21
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Table 13.
Results with different number of groupsgin weight-only post-training quantization results on non-uniform scalar quantizationformat, without fine-tuning to the end-to-end loss. Wiki2 and C4 denotes perplexity on WikiText2 and C4, respectively, which aremeasured with the context size of 4096.
Number ofLlama-2-7B Llama-2-13B Llama-2-70B
Method groups g Bits# Wiki2# C4# Bits# Wiki2# C4# Bits# Wiki2# C4#
Original  16 5.12 6.63 16 4.57 6.05 16 3.12 4.97
LNQ  2.01 23.31 26.71 2.01 8.78 11.80 2.01 5.23 7.31LNQ + GuidedQuant 1 2.01 9.00 11.35 2.01 7.32 9.29 2.01 5.11 7.062 2.01 8.82 11.20 2.01 7.18 9.22 2.01 5.04 7.044 2.01 8.83 11.15 2.01 7.26 9.17   
LNQ  3.03 5.89 7.74 3.02 5.02 6.68 3.01 3.50 5.31LNQ + GuidedQuant 1 3.03 5.55 7.23 3.02 4.92 6.49 3.01 3.46 5.272 3.03 5.57 7.22 3.02 4.92 6.49 3.01 3.47 5.274 3.03 5.57 7.22 3.02 4.91 6.49   
LNQ  4.05 5.26 6.82 4.04 4.67 6.17 4.03 3.20 5.04LNQ + GuidedQuant 1 4.05 5.21 6.75 4.04 4.65 6.14 4.03 3.20 5.032 4.05 5.22 6.75 4.04 4.65 6.14 4.03 3.20 5.034 4.05 5.21 6.75 4.04 4.65 6.14   
Table 14.
Ablation study on optimizing discrete assignmentPin Problem(
8
). We compare two algorithms for optimizing discreteassignments; GPTQ and coordinate descent algorithm. Wiki2 and C4 denotes perplexity on WikiText2 and C4, respectively, which aremeasured with the context size of 4096.
OptimizationLlama-2-7B Llama-2-13B Llama-2-70B
Method method for P Bits# Wiki2# C4# Bits# Wiki2# C4# Bits# Wiki2# C4#
Original  16 5.12 6.63 16 4.57 6.05 16 3.12 4.97
LNQ + GQuant GPTQ 2.01 9.65 11.83 2.01 7.96 11.65 2.01 4.92 6.93Coordinate Descent 2.01 8.83 11.15 2.01 7.26 9.17 2.01 5.04 7.04
LNQ + GQuant GPTQ 3.03 5.58 7.25 3.02 4.91 6.50 3.01 3.47 5.27Coordinate Descent 3.03 5.57 7.22 3.02 4.91 6.49 3.01 3.47 5.27
LNQ + GQuant GPTQ 4.05 5.22 6.75 4.04 4.65 6.14 4.03 3.20 5.03Coordinate Descent 4.05 5.21 6.75 4.04 4.65 6.14 4.03 3.20 5.03
Llama-2-7B and Llama-2-13B, andg = 2for Llama-2-70B. Still, smaller number of groups are sufficient for achievingmost of the performance gains, making them a practical choice for resource-constrained scenarios.D.6. Ablation Study on Assignments Optimization in LNQIn this section, we evaluate our choice of using cyclic CD algorithm instead of GPTQ to solve Problem(
8
)for a fixedcodebookc(j)in LNQ. In particular, we compare two variants of LNQ with the GuidedQuant objective: the variant describedin Section
4.2
, which updates the assignments using cyclic CD, and alternative variant that uses GPTQ for assignmentsupdates. Both variants update the codebook using the closed-form solution in(
9
). We report the results on Llama-2-7Bmodel, evaluated on WikiText2 and C4 datasets, in Table
14
. Our experiments show that CD consistently outperforms ormatches GPTQ, validating our choice of using CD to optimize the assignment matrix P(j). D.7. End-to-end Fine-tuning ResultsRecent weight-only quantization methods have explored fine-tuning quantized models using extensive data and compute toimprove performance for low-bit models (
Tseng et al.
,
2024a
;
b
;
Malinovskii et al.
,
2024a
). In Table
15
, we summarize the
22
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Table 15.
Weight-only quantization results on Llama-2-7B model after fine-tuning with end-to-end loss. For scalar quantization methods,we report the performance after fine-tuning with PV-Tuning (
Malinovskii et al.
,
2024a
).
Method Bits# Wiki2# C4#
Type Original 16 5.12 6.63
Weight-onlyScalarGPTQ 2.14 8.43 10.82SqueezeLLM 2.01 6.78 8.82LNQ + GQuant (Ours) 2.01 6.53 8.53
SqueezeLLM 3.03 5.53 7.23LNQ + GQuant (Ours) 3.03 5.50 7.14
Table 16.
Weight-and-activation quantization results on Llama-2-7B model, while quantizing weights into 2- and 3-bits. Wiki2 denotesperplexity on Wikitext2 with the context size of 2048. WxAyKVzindicates quantizing weights intox-, activations intoy-, and KV cacheto z-bits, respectively.
Method Bits# Wiki2#
Original 16 5.12
SpinQuant W2A4KV4 100.22SpinQuant + GQuant (Ours) W2A4KV4 36.05
SpinQuant W3A4KV4 6.61SpinQuant + GQuant (Ours) W3A4KV4 6.29
performance of quantized models after further fine-tuning on end loss using more data and compute for scalar weight-onlyquantization. We implement PV-Tuning (
Malinovskii et al.
,
2024a
) in non-uniform scalar quantization setting and report theperformance of both our model and SqueezeLLM after fine-tuning with it. For SqueezeLLM and LNQ + GuidedQuant, weobtain the results using the official open-source implementation of PV-Tuning. Our fine-tuning setup uses training data fromRedPajama dataset (
Computer
,
2023
), with a context size of 4096 tokens, a batch size of 128 sentences, and fine-tuning for128 steps in 2-bit quantization and 32 steps in 3-bit quantization. For GPTQ (uniform scalar quantization), we report theresults from the PV-Tuning paper (
Malinovskii et al.
,
2024a
).The results in Table
15
show that our method remains superior, though the gap narrows at larger bit-widths. We hypothesizethat existing PTQ methods, which rely on less accurate surrogate objectives, have smaller gaps at higher bit-widths, allowingfine-tuning to narrow the difference. However, in more extreme compression settings, where the gap is wider, our methodmaintains its advantage even after fine-tuning.D.8. Results on Smaller Bit-width in Weight-and-activation QuantizationIn weight-and-activation quantization, we further conduct an additional experiments with lower bit-widths for weights,specifically 2-bit and 3-bit, while keeping activations and KV caches at 4-bit precision (denoted as W2A4KV4 andW3A4KV4, respectively), on Llama-2-7B model. The results, shown in Table
16
, demonstrate that GuidedQuant outperformsbaseline methods by larger margin in these more extreme scenarios, highlighting the strength of our approach under stricterbit-width constraints.D.9. Comparison with mixed-precision variant of SqueezeLLMThe dense-and-sparse variant of SqueezeLLM (
Kim et al.
,
2024
), which preserves a small fraction of weights in 16-bitprecision to maintain accuracy, is orthogonal to our method and can be combined with it. Accordingly, in Table
17
, wereport results for SqueezeLLM, LNQ, and LNQ + GuidedQuant methods, with the dense-and-sparse approach applied to allof them, using the identical experimental setting with Table
3
. Following the original SqueezeLLM paper, we retain 0.45%of the weights in 16-bit and evaluate with 2-, 3-, and 4-bit quantization on the Llama-2-7B model. The results show thatLNQ with GuidedQuant consistently outperforms the baselines in the dense-and-sparse setting as well, demonstrating thesuperiority and robustness of our method.
23
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Table 17.
Weight-only scalar post-training quantization results on Llama-2-7B model, evaluated under a dense-and-sparse setting,preserving 0.45% of the weights in 16 bits. Wiki2 and C4 denotes perplexity on WikiText2 and C4, respectively. The perplexity ismeasured with the context size of 4096.
Method Bits# Wiki2# C4#
Original 16 5.12 6.63
SqueezeLLM (0.45%) 2.22 10.64 14.10LNQ (0.45%) (Ours) 2.22 8.26 10.34LNQ + GuidedQuant (0.45%) (Ours) 2.22 8.00 10.18
SqueezeLLM (0.45%) 3.24 5.58 7.23LNQ (0.45%) (Ours) 3.24 5.49 7.15LNQ + GuidedQuant (0.45%) (Ours) 3.24 5.48 7.12
SqueezeLLM (0.45%) 4.27 5.22 6.75LNQ (0.45%) (Ours) 4.27 5.20 6.74LNQ + GuidedQuant (0.45%) (Ours) 4.27 5.20 6.73
D.10. Results on Different QTIP Variants (1MAD, 3INST, HYB)The original QTIP paper introduced three variants of their method: 1MAD, 3INST, and HYB (
Tseng et al.
,
2024b
). Both1MAD and 3INST are look-up table-free methods, while HYB incorporates a small look-up table that fits within the L1cache of modern GPUs. The authors reported post-training quantization results without fine-tuning for the 1MAD and3INST formats, while quantization with fine-tuning was reported for the HYB format. To maintain consistency, we reportthe better-performing variant between 1MAD and 3INST in Table
4
for both QTIP and our method (QTIP + GuidedQuant).For completeness, the full performance results across 1MAD and 3INST are provided in Table
18
.It is worth noting that QTIP has only open-sourced the CUDA acceleration kernel for HYB, although it is theoreticallypossible to implement kernels for 1MAD and 3INST. Therefore, we also include the post-training quantization results(without fine-tuning) for the HYB format as well, summarized in Table
18
. The results show that the variations among QTIPmethods have minimal impact on the results, and our method consistently outperforms all others in Table
4
, regardless of theQTIP variant chosen.D.11. Discussion on Block-diagonal Fisher ApproximationIn this section, we review existing neural network compression methods that use a block-diagonal Fisher matrix approxi-mation of the Hessian and highlight their differences from GuidedQuant. In particular, we discuss WoodFisher (
Singh &Alistarh
,
2020
) for pruning CNNs, Optimal BERT Surgeon (
Kurtic et al.
,
2022
) for pruning BERT models, and BRECQ (
Liet al.
,
2021
) for quantizing CNNs.WoodFisher and Optimal BERT Surgeon use blocks of arbitrary sizeB fi Balong the diagonal to reduce the storage cost.WoodFisher exploresBsize off20; 100; 1000; 5000; 12288; 37000gin ResNet-20 (
He et al.
,
2015
), while Optimal BERTSurgeon usesB = 50, since the larger block size does not fit in the memory. BRECQ leaves the blocks that correspondto the parameters within each residual block in CNNs, and further uses a first-order Taylor approximation on the residualblock's outputs to estimate the second-order error for each block to avoid the need to handle prohibitively large matrices.The proposed GuidedQuant maintains the blocks corresponding to each output channel, resulting theBsize to be4096to11008for Llama-2-7B model. Directly computing these block-diagonal matrices would be infeasible, requiring over110TB for and more than13; 000GPU hours on RTX 6000 Ada GPU for Llama-2-7B. To address this, GuidedQuant averagesthe Fisher diagonal blocks within each group, approximately preserving dependencies within each output channel at thescale of modern LLMs. We present the theoretical complexity of GuidedQuant in Section
3.2
, report its practical cost inTable
9
, and report the performance of approximating more (opting for smaller number of groups) in Section
D.5
.In Figures
3
and
4
, we illustrate submatrices of the scaled Fisher information matrix,nF(l)jfi 106, for the linear layers inthe first Transformer block of the Llama-2-7B model, alongside corresponding approximation results. Here,ndenotesthe number of calibration data, and the results are computed using calibration data from the RedPajama dataset, which
24
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Table 18.
Weight-only post-training quantization results on different QTIP variants (1MAD, 3INST, HYB), without fine-tuning to theend-to-end loss. Wiki2 and C4 denotes perplexity on WikiText2 and C4, respectively, which are measured with the context size of 4096.
Llama-2-7B Llama-2-13B Llama-2-70B
Variant Method Bits# Wiki2# C4# Wiki2# C4# Wiki2# C4#
Original 16 5.12 6.63 4.57 6.05 3.12 4.97
1MAD QTIP 2.00 7.05 9.14 5.59 7.46 3.87 5.70QTIP + GQuant (Ours) 2.00 6.11 7.99 5.33 7.05 3.80 5.61
QTIP 3.00 5.38 6.99 4.74 6.28 3.27 5.09QTIP + GQuant (Ours) 3.00 5.28 6.87 4.71 6.22 3.25 5.08
QTIP 4.00 5.17 6.71 4.62 6.10 3.16 5.00QTIP + GQuant (Ours) 4.00 5.16 6.68 4.61 6.09 3.15 5.00
3INST QTIP 2.00 6.82 8.96 5.52 7.39 3.90 5.69QTIP + GQuant (Ours) 2.00 6.16 7.99 5.33 7.04 3.82 5.61
QTIP 3.00 5.40 7.01 4.74 6.28 3.27 5.09QTIP + GQuant (Ours) 3.00 5.30 6.87 4.70 6.22 3.26 5.08
QTIP 4.00 5.17 6.71 4.62 6.10 3.16 5.00QTIP + GQuant (Ours) 4.00 5.16 6.68 4.61 6.09 3.15 5.00
HYB QTIP 2.00 6.84 9.03 5.62 7.46 3.93 5.74QTIP + GQuant (Ours) 2.00 6.19 8.06 5.36 7.10 3.84 5.64
QTIP 3.00 5.39 7.03 4.76 6.31 3.28 5.10QTIP + GQuant (Ours) 3.00 5.32 6.89 4.72 6.24 3.27 5.09
QTIP 4.00 5.19 6.73 4.63 6.12 3.17 5.01QTIP + GQuant (Ours) 4.00 5.18 6.70 4.61 6.10 3.16 5.00
consists of 1024 sentences with 4096 tokens each. Since each linear layer in the model containsdinfi doutweights, fullyvisualizing its Fisher information matrix would yield a matrix of sizedindoutfi dindout, which is computationally prohibitive.Therefore, we restrict our visualization to the submatrix corresponding to the first two output channels of each layer. Sinceeach output channel hasdinweights, this results in visualizing a2dinfi 2dinmatrix. Within the Transformer block of theLlama-2-7B model, there are seven linear layers:self
attn.q
proj,self
attn.k
proj,self
attn.v
proj,self
attn.o
proj,mlp.gate
proj,mlp.up
proj, andmlp.down
proj. For the first six layers,din= 4096,so we visualize an8192 fi 8192matrix, while for the final layer (mlp.down
proj) withdin= 11008, an22016 fi 22016 matrix is visualized.We compare two approximation strategies:

WoodFisher: This approach retains the blocks size ofB fi Balong the diagonal. The storage requirement for thismethod is B dindout.

GuidedQuant: Here, the block size is set todinfi dinand blocks are averaged within groups. This strategy requiresg d2instorage, where g is the number of groups.To ensure a fair comparison, we choose the WoodFisher block size asB = dg dout=dine. Specifically, we chooseg = 4for the GuidedQuant, which results inB = 4for the self-attention projection layers,B = 2for themlp.gate
projandmlp.up
proj layers, and B = 11 for the mlp.down
proj layer.The visualizations reveal that the original Fisher information matrix exhibits strong off-diagonal values and a prominentblock-diagonal structure with blocks of sizedinfi din. This indicates stronger interactions among weights within the sameoutput channel compared to those across different channels. Overall, the GuidedQuant approximation captures significantlymore of this structural detail than the WoodFisher-style block-diagonal approximation, which retains only arbitrarily sizeddiagonal blocks.
25
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Fisher (Original) WoodFisher GuidedQuant (Ours)
self
attnq
proj
self
attnk
proj
self
attnv
proj
self
attno
proj
Figure 3.
Visualization of the scaled Fisher information matrix,nF(l)jfi 106, for the first two output channels in theself
attn.q
proj,self
attn.k
proj,self
attn.v
proj, andself
attn.o
projlayer of the first Transformer block in Llama-2-7B model.Left: the original Fisher matrices; Middle: the WoodFisher style block-diagonal approximation (block sizeB = 4for all of the layers);Right: the GuidedQuant approximation (the number of groupsg = 4). Both approximations are compared under an equal storage budget.
26
GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
Fisher (Original) WoodFisher GuidedQuant (Ours)
mlpgate
proj
mlpup
proj
mlpdown
proj
Figure 4.
Visualization of the scaled Fisher information matrix, nF(l)jfi 106, for the first two output channels in the mlp.gate
proj, mlp.up
proj, andmlp.down
projlayer of the first Transformer block in Llama-2-7B model. Left: the original Fisher matrices;Middle: the WoodFisher style block-diagonal approximation (block sizeB = 2,B = 2, andB = 11, respectively); Right: theGuidedQuant approximation (the number of groups g = 4). Both approximations are compared under an equal storage budget.
27
