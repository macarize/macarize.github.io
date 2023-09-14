---
layout: distill
title: CAM guided pedestrian attribute recognition
description: 저의 졸업논문인 CAM-PAR에대한 간략한 소개 입니다.
giscus_comments: true
date: 2023-09-10
featured: true
importance: 1
category: CAM-PAR
img: assets/img/cam_par_intro/full_framework.png

authors:
  - name: Hyo jeong lee

bibliography: 2018-12-22-distill.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: What is Pedestrian Attribute Recognition(PAR)?
    # if a section has subsections, you can add them as follows:
    # subsections:
  - name: Pedestrian Attributes
  - name: Applications
  - name: Challenges
    subsections:
        - name: Class/Label imbalance
        - name: Low-resolution
        - name: Lable correlation
  - name: Method
    subsections:
        - name: CAM-PAR
        - name: CFAR
  - name: Experiments
    subsections:
        - name: Evaluation Metrics and Datasets
        - name: Results
        - name: Ablation Study

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }

---
<!-- ---
layout: page
toc:
  sidebar: left
title: CAM guided pedestrian attribute recognition
description: 저의 졸업논문인 CAM-PAR에대한 간략한 소개 입니다.
# img: assets/img/12.jpg
importance: 1
category: work
--- -->


Pedestrian Attribute Recognition(PAR)는 보행자 이미지에서 여러 속성을 찾아내는 컴퓨터 비전 문제 중 하나입니다.
이 페이지에서는 PAR문제에 대한 정의, 특징, 실생활에서의 적용 범위 그리고 제가 연구개발한 PAR 모델의 간략한 아웃라인을 소개하려고 합니다. 

## What is Pedestrian Attribute Recognition(PAR)?
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cam_par_intro/PA100K.png" title="Sample of PA100K dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig.1 보행자 이미지 데이터셋 중 하나인 PA100K 중 이미지 샘플들. 첫 번째 이미지의 보행자 속성은 'female', 'hand-carry' 그리고 'short-sleeves'등이 될 수 있다. 
</div>
보행자 속성 인식 문제(Pedestrian Attribute Recognition, PAR)는 보행자 이미지가 주어졌을 때, 그 안에서 여러 종류의 보행자 속성을 검출해 내는것을 목표로 하고 있습니다.
조금 더 명확하게 표현하자면, PAR 문제는 
보행자 이미지 $$I$$ 에서 해당 보행자의 속성을 제일 잘 표현하는 속성 그룹 $$a_i$$을 사전에 정의된 속성 집합 $$A = {a_1, a_2, ..., a_L}$$으로부터 구하는 것을 목표로 합니다.

## Pedestrian Attributes
보행자 속성의 종류로는 의상(i.e., shorts, skirts), 헤어스타일(i.e., bald, long-hair) 등의 이미지 내의 집약적인 부분의 시각적 이미지로부터 알아낼 수 있는 속성들 또는 나이(i.e., middle-age)나 역할(i.e., clark)과 같이 추상적인 개념들 그리고 행동(i.e., walking, running), 다른 사물 또는 사람들과의 관계(i.e., picking, calling)를 나타내는 속성 등이 있을 수 있습니다. 

## Applications
PAR은 다른 컴퓨터 비전 문제의 하위 문제로 기여를 할 수 있으며, 실생활 문제에서도 깊이 관여되고 있습니다. 예를 들어, 횡단보도에서의 Scene-understanding 문제에서 PAR은 해당 장면을 이해하는데 중요한 역할을 수행합니다. 실생활 문제에서는 감시 시스템에서 PAR 모델이 자주 사용됩니다. CCTV에 PAR을 접목시켜 움직임이 수상한 인물을 검출해 내거나, 가정에서의 위급 상황에 대처할 수 있는 정보를 주기도 합니다. 최근 국내에서의 흉악범죄 발생으로 치안 강화에 대한 요구가 증가하며, PAR 기술의 역할이 더욱 중요해지는 상황입니다.

## Challenges
Multi-label classification의 하위 문제인 만큼, PAR문제는 이와 비슷한 챌린지를 공유하고 있음과 더불어 모델 성능을 높이는데에 걸림돌이 되는 고유의 챌린지들도 가지고 있습니다. Multi-label classification 문제와 공유하고 있는 챌린지 중 가장 대표적인 챌린지는 데이터의 분포가 한쪽으로 치우쳐 있는 imbalanced data distribution 문제가 있습니다. 또한, PAR 고유의 챌린지로 볼 수 있는 것 들로는, low-resolution 문제와 보행자 속성간의 상관관계 문제가 있습니다. 

### Class/Label imbalance
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cam_par_intro/pa100k_plot.png" title="Sample of PA100K dataset" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cam_par_intro/sorted_pa100k_plot.png" title="Sample of PA100K dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig.2 PA100K 데이터셋의 레이블 분포를 나타내는 그래프. x-축은 속성의 인덱스이고 y-축은 속성이 학습 데이터에서 나타나는 횟수를 나타낸다. 오른쪽은 등장 횟수가 빈번한 속성부터 내림차순으로 정렬한 그래프이다. long-tail distribution의 형태를 보인다. 
</div>
심하게 불균형한 학습 데이터는 모델이 비중을 많이 차지하고있는 속성에 치우쳐 학습하고, 적은 비중으로 등장하는 속성에 대해 충분한 학습을 하지 못하게 된다는 문제점이 있습니다. 예를 들어, 보행자 속성 데이터셋의 종류 중 하나인 RAPv1에서는 거의 대부분의 데이터셋 샘플들이 검정색 머리를 가지고 있습니다. Label imbalance를 해결하지 않고 해당 데이터셋을 사용해 모델을 학습한다면, 검정 머리가 아닌 사람에 대해서도 검정 머리로 분류할 가능성이 높아집니다. 이를 해결하기 위해, Over/under sampling, hard negative sample mining과 같은 샘플링 기반 방법들과 focal Loss, weighted loss 같은 알고리즘 기반 방법들이 소개가 되었습니다.

### Low-resolution
현재까지 공개되어있는 PAR 문제를 위한 데이터셋들 대부분은 감시 카메라에서 얻어낸 사진들입니다. 이러한 이유로, 학습 데이터의 화질이 낮은 문제가 있습니다. 학습 데이터의 화질이 낮을 시 발생할 수 있는 문제점으로는, 작은 크기를 가지고 있는 속성들(fine grainded attributes)을 모델이 학습하기 어렵다는 것 입니다. 이를 해결하기 위해, PAR 모델들은 낮은 화질의 이미지에도 강건한(robust)한 특성을 가져야 합니다.

### Lable correlation
많은 선행 연구들에서, 보행자 속성들은 서로 강한 관계를 가지고 있다는 것을 알 수 있습니다. 예를 들어, 치마를 입은 사람일 경우 성별이 남자일 확률보다 여자일 확률이 높습니다. 이러한 관계 정보는 모델 학습에 중요한 요소가 될 수 있으며, 이를 효과적으로 활용하는것이 PAR 연구의 한 주요 챌린지로써 연구되고 있습니다.

## Method
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cam_par_intro/full_framework.png" title="Sample of PA100K dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig.3 제안하는 모델의 전체 프레임워크. 실선 화살표는 첫 번째 학습 과정이며 그 중 주황색 화살표는 disentanglement 과정을 나타낸다. 보라색 점선 화살표는 두 번째 학습 과정이며, 여기서 속성집합-관계 정보를 이용한다. 
</div>
저희는 위 PAR 문제에서의 챌린지들을 해결하기 위해, class activation map(CAM)을 활용한 PAR 모델을 제안합니다. 저희가 제안한 모델은 크게 두 모듈로 나누어 생각할 수 있습니다.
<ul>
    <li>CAM-PAR: CAM을 사용한 Feature disentanglement 수행 후 여러개의 개별 Classifier를 사용하여 속성 검출.</li>
    <li>CFAR: 속성간의 관계를 학습한 Collaborative Filtering(CF) 추천 시스템을 사용해 예측한 속성 값들을 보정.</li>
</ul>

### CAM-PAR
#### Previous works - DAFL
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cam_par_intro/ofmc.png" title="Sample of PA100K dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    (a) One-Feature-for-Multiple-Attributes(OFMA) mechanism
</div><div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cam_par_intro/Ofom.png" title="Sample of PA100K dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    (b) One-Feature-for-One-Attributes(OFOA) mechanism 
</div>
<div class="caption">
    Fig.4 OFMA, OFOA 메카니즘의 일반적 파이프라인.
</div>
CAM-PAR의 경우, disentangled attribute feature learning (DAFL) 논문을 기반으로 한 feature disentanglement 기법입니다. DAFL 논문에서, 저자는 대부분의 PAR 모델이 따르고 있는 One-Feature-for-Multiple-Attributes(OFMA) 메커니즘의 한계를 강조합니다. OFMA 메커니즘은 하나의 공유된 feature cector에서 여러개의 속성을 검출하는 메커니즘을 말하며, 이를 따를 시 multi-label classification 상황에서 모델의 강건함(robustness)을 수학적으로 기대하기 어렵게 됩니다. 이를 해결하기 위해, DAFL 에서는 One-Feature-for-One-Attributes(OFOA) 메커니즘을 제안합니다. OFOA 메커니즘은 하나의 공유된 feature vector를 속성 갯수만큼의 속성-특화된(attribute specific) feature vector들로 disentangle 하여 속성을 검출합니다. Disentangle을 수행하는 일은 semantic spatial cross-attention modules(SSCA)이 수행하게 됩니다. SSCA는 다수의 순차적인 attention 모듈을 사용하여 속성-특화 벡터들을 학습합니다. 이 방법은 여러 벤치마크들에서 좋은 성과를 내었지만, disentangle에 추가적인 학습할 파라미터들이 추가된다는 단점이 있습니다. 또한, 학습하려는 데이터셋에 알맞는 SSCA 모듈의 갯수를 실험적으로 찾아내야한다는 단점도 존재합니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cam_par_intro/OFOA.png" title="Sample of PA100K dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig.5 DAFL의 SSCA 모듈을 사용한 OFOA 메카니즘
</div>

우리는 disentanglement에 사용되는 SSCA 모듈의 단점들을 보안한 class activation map guided pedestrian attribute recognition(CAM-PAR)를 제안합니다. CAM-PAR는 disentangle 과정에 추가적인 학습 파라미터들이 존재하지 않으며, 연속적인 모듈이 필요하지 않습니다. CAM-PAR의 대략적인 파이프라인은 다음과 같습니다.
<ul>
    <li>각 속성에 대한 class activation map 생성</li>
    <li>더욱 정확한 CAM 생성을 위한 feature fusion 진행</li>
    <li>생성된 CAM을 global average pooling(GAP)을 진행하기 직전의 제일 마지막 인코더 레이어 feature map과 element-wise multiplication. </li>
</ul>
1번 과정으로, 우리는 보행자 이미지에서 각 속성을 대표하는 부분의 activation map을 얻을 수 있습니다. 기본적인 CAM과 같이 제일 마지막 인코더 레이어로부터만 CAM을 생성하는것이 아닌 백본 인코더의 여러 깊이의 레이어에서 CAM을 생성하게 되는데(i.e., 2nd, 3rd and 4rd layer of ResNet50), 이는 다음 과정인 feature fusion을 하기 위함입니다. 2번째 과정에서 우리는 feature fusion을 이용하여 얻은 CAM을 개선합니다. "Shallow Feature Matters for Weakly Supervised Object Localization (CVPR 2021)" 논문에서, 저자는 얕은 층의 CNN encoder feature으로부터 생성된 CAM이 가지고 있는 세밀한 경계선 정보를 강조합니다. 하지만, 얕은 층의 encoder feature는 단독으로 사용하기에는 background noise가 심한 단점이 있습니다. 이러한 노이즈를 제거하기 위해, 위 논문의 저자는 multiplication based channel attention(MCA) 모듈을 사용하여 해결하였습니다. MCA 모듈은 여러 층에서의 feature vector를 하나로 합친 뒤 다시 나누는 작업을 진행하며, MCA 모듈을 통과한 feature vector들은 multiplicative feature fusion(MFF)를 거쳐 하나의 feature vector로 만들어지게 됩니다. 우리는 이 MCA 모듈과 MFF 전략을 빌려와 최종적으로 feature disentanglement를 수행하는데 쓰일 CAM의 정확도를 높입니다. 마지막으로, 정제된 CAM과 global average pooling(GAP)을 진행하기 직전의 제일 마지막 인코더 레이어 feature map과 element-wise multiplication을 진행하여 속성-특화 벡터들을 얻습니다.

### CFAR
#### CF as Auxiliary Information
추천 시스템(Recommender System)의 한 종류인 Collaborative Filtering(CF)방법은 "비슷한 행동을 보이거나 어떤 아이템에 대해 비슷한 평가를 내린 유저 그룹 $$A$$와 $$B$$가 있을 때, 다른 새로운 아이템에 대해서도 두 그룹이 비슷한 평가나 행동을 보일 것"이라는 전제를 바탕으로 유저에게 아이템을 추천하는것을 목표로 합니다. 이 CF방법은 접근법에 따라 크게 세 가지의 카테고리로 나눌 수 있습니다. 

<ul>
    <li>Memory-based</li>
    <li>Model-based</li>
    <li>Hybrid recommender systems</li>
</ul>

이 중, 우리는 singular value decomposition(SVD)를 사용해 잠재적인 유저 피드백(e.g., clicks, purchases)을 잘 설명할 수 있는 latent factor들을 학습하는 방법을 사용하여 PAR 문제에서의 속성집합의 분포를 모델링 하고 이를 속성 예측의 부가적 정보로 활용하는 모듈인 CFAR을 제안합니다. CFAR에서 보행자 이미지를 추천 시스템에서의 user, 속성들을 아이템으로 생각하여 첫 번째 보행자 속성 검출 단계에서 찾지 못한 속성들을 collaborative filtering을 사용하여 찾고자 합니다.

Input image $$x$$와 속성 집합 $$y$$가 있을 때, $$j-th$$속성이 $$i-th$$ 이미지 $$x_i$$에 존재할 확률를 $$r_{i, j} = \sigma(logit_{i, j})$$ 라고 할 때 ($$logit_{i, j}$$는 classifier output 값), 우리는 논문 'Collaborative Filtering for Implicit Feedback Datasets'에서 제안된 방법을 따라 latent factor를 학습합니다. Latent factor를 학습하기 위해, 아래의 비용 함수(cost function)을 최소화 합니다. 아래 수식에서, $$x_f \in R^{N \times D}$$는 이미지-팩터(image-factor), $$y_f \in R^{M \times D}$$는 속성-팩터(attriute-factor)들을 나타내며 $$N, M$$은 순서대로 이미지의 수, 속성의 수를 나타냅니다.

$$
\begin{equation}
\label{eq:bin}
p_{i, j} = 
\begin{cases}
    1 & \text{ if } r_{i, j} \geq p^t  \\ 
    0 & \text{ if } r_{i, j} < p^t \\
\end{cases}
    ,\qquad c_{i, j} = 1 + \alpha r_{i, j}
\end{equation}
$$

\begin{equation}
\label{eq:ALS_par}
    min(x, y) \sum_{i, j} c_{i,j}(p_{i, j} - x^T_{f_i}y_{f_j})^2 + \lambda(\sum_i\left\|x_{f_i}\right\|^2 + \sum_j\left\| y_{f_j}\right\|^2)
\end{equation}

(1)번 식에서, 우리는 classifier output $$r_{i, j}$$를 임계값 $$p^t$$를 기준으로 이진화합니다. 이 이진화된 값 $$p_{i, j}$$는 $$j$$번째 속성이 $$i$$번째 이미지에서의 존재유무를 가리킵니다. 그리고 이 $$p_{i, j}$$의 신뢰도(confidence)를 $$c_{i, j}$$의 형태로 나타냅니다. 
(2)번 식을 살펴볼 때, 알려진 실제 속성 $$j$$가 이미지 $$i$$에 존재할 확률 $$p_{i, j}$$ 와 이미지-팩터와$$x_{f_i}$$ 속성-팩터$$y_{f_j}$$간의 오류를 최소화하며 latent factor들을 학습함을 알 수 있습니다. 이 과정에서 학습된 latent factor들은 다음 단계의 학습과 최종 추론 단계에서 사용됩니다.

#### Attribute-set Corrfelation Dictionary
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cam_par_intro/dictionary.png" title="Sample of PA100K dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig.6 Attribute-set correlation dictionary의 파이프라인.
</div>
위 latent factor들은 학습 데이터셋을 사용해 배우게 됩니다. 일반적으로 collaborative filtering 방식에서 새로운 유저와 그의 선호 아이템이 추가 될 경우, 모델이 이 새로운 유저에 대한 추천을 실행하기 위해서 추가적인 업데이트가 필요합니다. 우리의 PAR 문제 아래에서, 새로운 유저와 그 유저의 선호 아이템은 테스트 이미지와 그 이미지 내에 존재하는 속성들로 바꾸어 생각됩니다. 테스트 이미지마다 CF모델을 지속적으로 업데이트하는 일은 비용적으로 소모가 크며, 우리는 이를 해결하기 위해 Attributed-set correlation dictionary를 제안합니다.

첫 번째 학습 과정의 logit들이 있을 때, 우리는 k-means 클러스터링을 진행하여 $K$개의 centroid들을 딕셔너리 키 $$p^k \in R^{K \times D}$$로 지정합니다. 또한, 각 $$p^k$$과 같은 클러스터에 속한 logit들에 대해 CF를 진행하였을 때 얻어진 신뢰도값의 평균을 해당 딕셔너리 키에 대한 밸류 값으로 설정합니다. 

두 번째 학습 과정에서, 첫 번째 classifier에서 나온 logit과 제일 가까운 키 $$p^k$$의 값 $$\hat{r}$$과 최초의 logit값을 하나의 벡터로 이어붙여 두 번째 fully connected classifier $$fc_2$$에 전달합니다. 이를 통해 손실값을 계산 한 후 $$fc_2$$를 학습합니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cam_par_intro/set_relation.png" title="Sample of PA100K dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

속성-속성 일대일 관계를 고려하는 다른 연구들과 달리, 우리의 CFAR는 속성집합-속성집합의 관계를 모델링합니다. 이를 통해 더욱 풍부하고 정확한 속성간의 관계를 PAR 문제에 이용할 수 있습니다. 예를 들어, 첫 번째 상황 "carrying"과 "plastic bag" 속성이 존재할 때 "customer"속성이 존재할 확률이 놓습니다. 하지만, 두 번째 상황 "carrying"과 "baby stroller"가 존재할 때, 앞선 속성들과 "customer"는 유의미한 관계를 가지고 있지 않을 가능성이 높습니다. 속성-속성 관계만을 고려할 때, 두 번째 상황에서 속성 "carrying"의 존재는 "baby stroller"의 존재 여부와 무관하게 "customer" 속성의 존재 확률을 잘못 높힐 수 있습니다.

## Experiments
### Evaluation Metrics and Datasets
이 섹션에서는, 우리가 제안한 방식의 효과를 실험으로 나타내기 위해 사용한 evaluation metric들과 데이터셋을 소개합니다.

모델의 성능을 평가하는 척도로는, accuracy(Accu), precision(Prec), recall(Recall), F1 그리고 mean accuracy(mA)를 사용하였습니다. 모델의 학습과 평가에 사용된 데이터셋으로는 PA100K, RAPv1을 사용하였습니다.

### Results
<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cam_par_intro/comp_prev.png" title="Sample of PA100K dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig.7 이전 state-of-the-art 방법들과의 비교.
</div>

위의 표는 Baseline 모델로 삼은 strong baseline과 다른 이전 모델과의 성능을 비교한 표 입니다. 우리의 모델은 대부분의 SOTA 모델들의 성능을 뛰어넘었으며, baseline 모델보다 RAPv1에서 3.07%, PA100K에서 3.6%의 mA 향상을 보입니다. 그러나 모든 평가 기준에서 우리 방식과 같은 OFOA 메카니즘을 사용한 DAFL보다 낮은 성능을 보입니다. 이로, DAFL에서 채택한 다양한 손실함수들에 관해 우리의 모델에 적용하고 실험을 해 보는것이 필요합니다.

### Ablation Study
이 섹션에서, 우리는 우리가 제안한 모델의 세부 모듈들(CAM-PAR, feature fusion strategy, CFAR)의 효과를 Ablation Study로 검증합니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cam_par_intro/able.png" title="Sample of PA100K dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig.8 우리 방법의 부분들에 대한 RAPv1에 대한 실험
</div>

위 표에서, 베이스라인에 CAM-PAR를 적용하는것은 1.06%의 mA, 0.56%의 F1 점수의 향상을 보였습니다. 또한, feature fusion만을 적용하였을때는 0.83%의 mA, 1.61%의 F1 점수 향상을 보입니다. 베이스라인에 CAM-PAR과 feature fusion을 적용한 모델은 2.7%의 mA, 0.7%의 F1 점수 향상을 보였고, 마지막으로 CFAR까지 모든 방법을 적용하였을 때, 3.6%의 mA, 1.0%의 F1 점수 향상을 이뤘습니다.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/cam_par_intro/robust.png" title="Sample of PA100K dataset" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Fig.9 RAPv1 데이터셋에서, ground truth가 possitive인 attribute들에 대해 각 속성에 해당하는 classifier와 feature vector간의 각도를 나타낸 box plot. 
</div>
이에 더해, 우리는 CAM-PAR가 baseline 모델보다 더욱 강건함을 검증합니다. 여기서, 모델의 강건함을 나타내는 척도로 우리는 classifer weight과 feature vector간의 각도 $$\theta$$를 선택하였고, decision boundary의 각도인 90도에 멀리 $$\theta$$가 위치 할 경우 모델이 강건하다고 말할 수 있습니다. 위 box plot은 ground truth가 possitive인 attribute들에 대해 각 속성에 해당하는 classifier와 feature vector간의 $$\theta$$값을 나타냅니다. 빨간색으로 표시된 것은 baseline 모델이며, 파란색으로 나타낸 것은 우리가 제안한 모델입니다. 위에 위치한 녹색 가로 점선은 decision boundary이고, 아래에 위치한 녹색 가로 점선은 optimal angle입니다. 그림에서 볼 수 있듯이, 파란색으로 표시된 우리 모델의 $$\theta$$값이 baseline보다 decision boundary에서 멀리 떨어져 0도에 가깝게 위치하고있음을 볼 수 있습니다. 우리는 이를 통해 우리의 모델이 baseline 모델보다 더욱 강건하다고 말할 수 있습니다.