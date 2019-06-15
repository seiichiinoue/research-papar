## 0. どんなものか

- Latent codeを用いてGANによる生成画像をコントロールする。
- MNIST画像での検証では、latent codeを変えることで、手書き数字の種類や幅、傾きなどが変えられることが示されていた。

![](https://cdn-images-1.medium.com/max/1600/1*kyyjNnuNaOscjucBpql2AA.png)

## 1. 先行研究と比べて何がすごいか

潜在空間をコントロールすることを教師なしで効率よく行なったこと。

## 2. 技術や手法のキモ

Generatorの入力のzをc, zに分解した構造化されたノイズを用いて学習させる。その際、普通のGANでは、P(x|c)=P(x)を満たす解を見つけるとlatent codeを無視しするので、latente code cとrandom noise zの相互情報量を最大化しながら学習する。(GANの目的関数に制約項としていれる)

## 3. どうやって有効だと検証したか

相互情報量の下側限界の学習塾時での推移を通常のGANと比較していた。

![](https://cdn-ak.f.st-hatena.com/images/fotolife/n/nogawanogawa/20181113/20181113154528.jpg)

## 4. 議論はあるか

Latent codeの事後分布の算出を省くために定数として学習を行なっていた点。
分布を仮定して学習を行うとどのような変化が起こるかわからなかった。

## 5. 次に読むべき論文

- Y. Bengio, A. Courville, and P. Vincent, “Representation learning: A review and new perspectives,” Pattern Analysis and Machine Intelligence, IEEE Transactions on, vol. 35, no. 8, pp. 1798–1828, 2013.

- D. P. Kingma and M. Welling, “Auto-encoding variational bayes,” ArXiv preprint arXiv:1312.6114, 2013.