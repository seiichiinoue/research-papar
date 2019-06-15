# Least Squares Generative Adversarial Networks

- [arxiv](https://arxiv.org/abs/1611.04076)

## 0. どんなものか

- DCGAN（Deep Convolutional Layerを用いて敵対的入力の学習をさせるもの）の誤差関数をシグモイドクロスエントロピー誤差から，最小二乗誤差に変更したもの．
- 最小二乗誤差にすることによって
    - 生成される画像のクオリティが高くなる
    - 学習過程で安定したパフォーマンスが得られる

## 1. 先行研究と比べて何がすごいか

- 誤差関数の変更による精度の向上

## 2. 技術や手法のキモ

- 正規化GANとは違って，決定境界の正しい側に分類されているfake dataに対してもペナルティーを課すため，生成器は決定境界に向かってサンプルを生成することになる．
- 勾配消失の問題も解決する

![](https://cdn-images-1.medium.com/max/1600/1*aqBljk_YbAtmWpx81a1j_A.png)

## 3. どうやって有効だと検証したか

- LSGANの目的関数を最小化することはピアソンのX^2発散が最小化されることであると示している

## 4. 議論はあるか

- hyper paramの-1, 1, 0 || 0, 1, 1

## 5. 次に読むべき論文

- [info GAN](https://arxiv.org/abs/1606.03657)
- [conditional GAN](https://arxiv.org/abs/1411.1784)