# 機械学習における自然言語処理技術
## Embedding

### one-hot表現
- 最もシンプルなembedding手法．
- 表現したい語彙をリストに表現して，各単語を表現する次元を準備する．**表現したい文章に含まれているか否か**のベクトルで表現する．

### Word2Vec
- 大量のテキストデータを解析して，各単語の意味をベクトルで表現する方法．
- その中でもSkip-Gramモデル(ある単語の周辺に出現する単語の出現確率を計算する)が主に使われる
- Skip-Gram は２層のニューラルネットワークであり隠れ層は一つだけ．隣接する層のユニットは全結合している．

<img src="https://camo.qiitausercontent.com/135690f9499a9717cd0537dab997cc6bcd33f1a0/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f37373037392f36333338393465382d366531652d633237342d613333612d3833643062343665393961382e706e67" width=300>

- 目的関数を設定して，2層のニューラルネットワークを構築するが，**word2vecにおいて必要なものは，モデル自体ではなく隠れ層の重み**であることに注意．
- 入力としてある単語，出力にその周辺単語を与えてニューラルネットワークを学習させることで，「**意味が近い(=意味ベクトルの距離が近い)時は周辺単語の意味ベクトルもまた距離が近いはず**」という仮説に基づいたembedding表現を得ることができる．

<img src="https://camo.qiitausercontent.com/f087da40cde3cba6e8e4f49105f1d87fdacb3c9c/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f37373037392f64346335363430632d316465302d333731652d396465632d6435376637306636393863622e706e67" width=600>

<img src="https://camo.qiitausercontent.com/00f2cdda20a9a7d7852498621d48d4799ed728be/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f37373037392f63393539373939392d656166632d343635312d306135372d3830336434383039343036372e706e67" width=600>

- 上図のように，one-hotベクトルを入力として与えてあげれば，実際に対象の単語ベクトルを抽出する際は内積ではなくインデックスを使って抽出すれば良いだけなので単語数や隠れそうの次元を気にすることなくモデルを構築することができる

<img src="https://camo.qiitausercontent.com/96a336fd510e0e4100396787d80e2a4c199dca94/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f37373037392f64373435303531352d343064302d306662332d643536302d6639663864653466373036362e706e67" width=650>

- 出力層は上図のよう．対象の単語の重みベクトルを得たあと，中間層から出力層の単語の重みベクトルとの内積をとっている．つまり単語同士の内積が出力となっていることがわかる．



## LSTM

- LSTM(Long short-term memory)は，RNN(Recurrent Neural Network)の拡張として1995年に登場した，時系列データ(sequential data)に対するモデル，あるいは構造(architecture)の1種．その名は，Long term memory(長期記憶)とShort term memory(短期記憶)という神経科学における用語から取られている．LSTMはRNNの中間層のユニットをLSTM blockと呼ばれるメモリと3つのゲートを持つブロックに置き換えることで実現されている．

### Hochreiterの勾配消失問題
- 当時のRNNの学習方法は，BPTT(Back-Propagation Through Time)法とRTRL(Real-Time Recurrent Learning)法の2つが主流で，その2つとも完全な勾配(Complete Gradient)を用いたアルゴリズムだった
- しかし，このような勾配を逆方向(時間をさかのぼる方向)に伝播させるアルゴリズムは，多くの状況において「爆発」または「消滅」することがあり，結果として長期依存の系列の学習が全く正しく行われないといいう欠点が指摘されてきた
- Hochreiterは自身の修士論文(91年)において，時間をまたいだユニット間の重みの絶対値が指定の(ごくゆるい)条件を満たすとき，その勾配はタイムステップ$t$に指数関数的に比例して消滅または発散することを示した．
- これはRNNだけではなく，勾配が複数段に渡って伝播する深いニューラルネットにおいてもほぼ共通する問題らしい．

- 例えば，単体のユニット$u$から$v$への誤差の伝播について解析する．ステップ$t$における任意のユニット$u$で発生した誤差が$q$ステップ前のユニット$v$に伝播する状況を考えたとき，誤差は以下に示すような係数でスケールする．

$${\frac{\partial v_v (t-q)}{\partial v_u (t)} = 
\Biggl\{
\begin{array}{ll}
f'_v(net_v (t-1)) w_{uv} & q = 1 \\\
f'_v(net_v (t-q)) \sum_{l=1}^{n} \frac{\partial v_v (t-q+1)}{\partial v_u (t)}w_{lv} & q > 1
\end{array}
}$$

- $l_q=v$ と$l_0=u$を使用して、

$${\frac{\partial v_v (t-q)}{\partial v_u (t)} = \sum_{l_1 = 1}^{n} \cdots　\sum_{l_{q-1} = 1}^{n} \prod_{m=1}^q f'_{l_m} (net_{l_m} (t - m)) w_{l_m l_{m-1}}
}$$

- 上式より，以下の場合はスケール係数は発散し，その結果としてユニット$v$に到着する誤差の不安定性により学習が困難になる．

$${\begin{array}{ll}
| f'_{l_m}(net_{l_m} (t - m)) w_{l_m l_{m-1}} | \;  > 1.0 & for\; all\; m
\end{array}
}$$

- 一方，以下の場合はスケール係数は$q$に関して指数関数的に減少する．

$${\begin{array}{ll}
| f'_{l_m}(net_{l_m} (t - m)) w_{l_m l_{m-1}} | \;  < 1.0 & for\; all\; m
\end{array}
}$$

- これらの問題を解決するために考案されたのがLSTM

### LSTMモデル
- $R$と$W$は重み行列

<img src="https://camo.qiitausercontent.com/7a6631efe2ef70321264f254c2df625ec3cbd3ec/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f36303936392f34383566363334642d396264322d326565332d393962322d3039363934653833316635352e706e67" width=500>

### LSTMの順伝播計算

$\bar{z}^t = W_z x^t + R_z y^{(t-1)} + b_z, \ z^t = g(\bar{z}^t)$

$\bar{i}^t = W_{in} x^t + R_{in} y^{(t-1)} + p_{in} \odot c^{t-1} + b_{in}, \ i^t = \sigma(\bar{i}^t)$

$\bar{f}^t = W_{for} x^t + R_{for} y^{(t-1)} + p_{for} \odot c^{t-1} + b_{for}, \ f^t = \sigma(\bar{f}^t)$

$c^t = i^t \odot z^t + f^t \odot c^{t-1}$

$\bar{o}^t = \sigma(W_{out} x^t + R_{out} y^{(t-1)} + p_{out} \odot c^t + b_{out}), \ o^t = \sigma(\bar{o}^t)$

$y^t = o^t \odot h(c^t)$

$s.t. \ \sigma(x) = sigmoid(x) = \frac{1}{1 + e^{-x}}, \\ g(x) = h(x) = tanh(x)$

### 逆伝播

<img src="https://camo.qiitausercontent.com/64c89c6204641d58c04d6ad391ad1f8e273c75a4/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f36303936392f62626331376330352d343937392d323934312d383636372d3463663066303233356663332e706e67" width=500>

$\delta y^t = \Delta^t + R_z^T \delta z^{t+1} + R_{in}^T \delta i^{t+1} + R_{for}^T \delta f^{t+1} + R_{out}^T \delta o^{t+1} $

$\delta o^t = \delta y^t \odot h(c^t) \odot \sigma'(\bar{o}^t)$

$\delta c^t = \delta y^t \odot o^t \odot h'(c^t) + p_{out} \odot \delta o^t + p_{in} \odot \delta i^{t+1} + p_{for} \odot \delta f^{t+1} + \delta c^{t+1} \odot f^{t+1}$

$\delta f^t = \delta c^t \odot c^{t-1} \odot \sigma'(\bar{f}^t)$

$\delta i^t = \delta c^t \odot z^t \odot \sigma'(\bar{i}^t)$

$\delta z^t = \delta c^t \odot i^t \odot g'(\bar{z}^t)$



## Attention

<img src="https://camo.qiitausercontent.com/dc9367114569469784d8f5a92b23f3d4291a562f/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f36313037392f37633230383137622d336563632d363430322d666266362d6263303035643430383734392e706e67" width=700>

- Attentionの基本は$query$と$memory$($key$, $value$)．
- Attentionとは$query$によって$memory$から必要な情報を選択的に引っ張ってくること．$memory$から情報を引っ張ってくるときには， $query$は$key$によって取得する$memory$を決定し，対応する$value$を取得する．


### Encoder-Decoderにおけるattention
- 一般的なEncoder-Decoderの注意はエンコーダの隠れ層を$Source$，デコーダの隠れ層を$Target$として次式によって表される．

$$Attention(Target,Source)=Softmax(Target⋅Source^T)⋅Source$$

- より一般化すると$Target$を$query$(検索クエリ)と見做し，$Source$を$Key$と$Value$に分離する．

$$Attention(query,Key,Value)=Softmax(query⋅Key^T)⋅Value$$

<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20171221/20171221163903.png" width=500>

- この時$Key$と$Value$は各$key$と各$value$が一対一対応するkey-valueペアの配列，つまり辞書オブジェクトとして機能する．
- $query$と$Key$の内積は$query$と各$key$の類似度を測り，$softmax$で正規化した注意の重み (Attention Weight) は$query$に一致した$key$の位置を表現する．注意の重みと$Value$の内積は$key$の位置に対応する$value$を加重和として取り出す操作である．
- つまり注意とは$query$(検索クエリ)に一致する$key$を索引し，対応する$value$を取り出す操作であり，これは辞書オブジェクトの機能と同じである．例えば一般的な Encoder-Decoder の注意は，エンコーダのすべての隠れ層 (情報源)$Value$から$query$に関連する隠れ層 (情報)$value$を注意の重みの加重和として取り出すことである．

query の配列 Query が与えられれば，その数だけ key-value ペアの配列から value を取り出す．

### MemoryをKeyとValueに分離する意味
- key-valueペアの配列の初出は End-To-End Memory Network [Sukhbaatar, 2015] であるが，$Key$を Input，$Value$を Output (両方を合わせて Memory) と表記しており，辞書オブジェクトという認識はなかった．
- (初めて辞書オブジェクトと認識されたのは [Key-Value Memory Networks](https://arxiv.org/abs/1606.03126) [Miller, 2016] である．)

![](https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20171221/20171221164543.png)

- Key-Value Memory Networks では key-value ペアを文脈 (e.g. 知識ベースや文献) を記憶として格納する一般的な手法だと説明している．**$Memory$を$Key$と$Value$に分離することで$key$と$value$間の非自明な変換によって高い表現力が得られる**という．ここでいう非自明な変換とは，例えば「$key$を入力して$value$を予測する学習器」を容易には作れない程度に複雑な (予測不可能な) 変換という意味である．

- その後，言語モデルでも同じ認識の手法 [Daniluk, 2017] が提案されている．

![](https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20171221/20171221164537.png)

### attentionのweightの算出方法

<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20171221/20171221164307.png" width=500>

- 加法注意と内積注意があり，加法注意は一層のフィードフォワードネットワークで重みを算出する一方，内積注意はattentionの重みを$query$と$key$の内積で算出する．こちらは前者に比べてパラメータが必要ないため，効率よく学習ができる．

### self-attention

![](https://camo.qiitausercontent.com/e3841e989665ca207b2bafc5ae1bbb81074e5724/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f36313037392f30333366313339372d336361382d616464612d633066312d6530386661386630333031352e706e67)

- $input$($query$)と$memory$($key$, $value$)すべてが同じTensorを使うAttention
- Self-Attentionは言語の文法構造であったり，照応関係（its が指してるのは Law だよねとか）を獲得するのにも使われているなどと論文では分析されている

- 例えば「バナナが好き」という文章ベクトルを自己注意するとしたら，以下のような構造になる．

<img src="https://camo.qiitausercontent.com/c0357c70af7308f9be5bb30ad4e69fa2f7a00629/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f3132333538392f37353566343237302d336331302d653134612d343033362d3561626639306565663137312e706e67" width=500>

### Source-Target Attention

<img src="https://camo.qiitausercontent.com/4edacdcfaa0a7104ca91b11adbd85f3ea31c6ac6/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f36313037392f35343239393939302d623161382d393536392d636536342d3038393830366236303061622e706e67" width=600>

- Transformerではdecoderで使われる．

## Transformer
- 論文タイトルにもある通り，ATTENTION IS ALL YOU NEED．つまりRNNやCNNを使わずattentionのみを使用した機械翻訳タスクを実現するモデル．
- 元論文 [Attention is all you need](https://arxiv.org/abs/1706.03762)
- Google [プロジェクトページ](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
- Pytorch model [Github](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

![](https://camo.qiitausercontent.com/5af7348bde95e4f6c52da9c0f1a2c6a95a64510a/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f36313037392f65366532316335612d346432392d303333642d373731312d3834393337653235366332302e706e67)

- モデルの概要は以下の通り
	- エンコーダ: [自己注意, 位置毎の FFN] のブロックを6層スタック
	- デコーダ: [(マスキング付き) 自己注意, ソースターゲット注意, 位置毎の FFN] のブロックを6層スタック

- ネットワーク内の特徴表現は [単語列の長さ x 各単語の次元数] の行列で表される．注意の層を除いて0階の各単語はバッチ学習の各標本のように独立して処理される．

- 訓練時のデコーダは自己回帰を使用せず，全ターゲット単語を同時に入力，全ターゲット単語を同時に予測する．ただし予測すべきターゲット単語の情報が予測前のデコーダにリークしないように自己注意にマスクをかけている (ie, **Masked Decoder**)．評価/推論時は自己回帰で単語列を生成する．

- Transformerでは内積注意を縮小付き内積注意 (Scaled Dot-Product Attention) と呼称する．通常の内積注意と同じく$query$をもとにkey-valueペアの配列から加重和として$value$を取り出す操作であるが$Q$と$K$の内積をスケーリング因子$\sqrt{d_k}$で除算する．
- また，$query$の配列は1つの行列$Q$にまとめて同時に内積注意を計算する (従来通り$key$と$value$の配列も $K$,$V$ にまとめる)．

### 縮小付き内積注意
- 縮小付き内積注意は以下のように表される．

$$Attention(Q, K, V) = Softmax(\frac{Q K^T}{\sqrt{d_k}})V$$

- Mask (option) はデコーダの予測すべきターゲット単語の情報が予測前のデコーダーにリークしないように自己注意にかけるマスクである (Softmax への入力のうち自己回帰の予測前の位置に対応する部分を1で埋める)．

- Transformer では縮小付き内積注意を1つのヘッドと見做し，複数ヘッドを並列化した複数ヘッドの注意 (Multi-Head Attention) を使用する．ヘッド数$h=8$と各ヘッドの次元数$d_{model} / h=64$はトレードオフなので合計のパラメータ数はヘッド数に依らず均一である．

### 複数ヘッドの注意
- $d_{model}=512$次元の$Q,K,V$を用いて単一の内積注意を計算する代わりに，$Q,K,V$をそれぞれ$h=8$回異なる重み行列 $W^Q_i,W^K_i,W^V_i$ で $d_k,d_k,d_v=64$ 次元に線形写像して$h=8$個の内積注意を計算する．各内積注意の$d_v=64$次元の出力は連結 (concatenate) して重み行列$W_o$で$d_{model}=512$次元に線形写像する．

- 複数ヘッドの注意は次式によって表される．
<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20171221/20171221164416.png" width=500>
<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20171221/20171221164426.png" width=600>

### 位置毎のフィードフォワードネットワーク

- FFNは以下のように表される

<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20171221/20171221164438.png" width=350>

- $ReLU$で活性化する$d_{ff}=2048$次元の中間層と$d_{model}=512$次元の出力層から成る2層の全結合ニューラルネットワークである．


### 位置エンコーディング
- **TransformerはRNNやCNNを使用しないので単語列の語順(単語の相対的ないし絶対的な位置)の情報を追加する必要がある．**
- 本手法では入力の埋め込み行列(Embedded Matrix)に**位置エンコーディング(Positional Encoding)の行列$PE$を要素ごとに加算**する．
- 位置エンコーディングの行列$PE$の各成分は次式によって表される．

<img src="https://cdn-ak.f.st-hatena.com/images/fotolife/R/Ryobot/20171221/20171221164434.png" width=350>

- ここで$pos$は単語の位置，$i$は成分の次元である．位置エンコーディングの各次元は波長が$2 \pi$から$10000⋅2 \pi$に幾何学的に伸びる正弦波に対応する．

<img src="https://i.stack.imgur.com/zvol4.png" width=500>

- 横軸が単語の位置(0 ~ 99)，縦軸が成分の次元(0 ~ 511)，濃淡が加算する値(-1 ~ 1)．



## BERT

### 概要
- 単語の分散表現を獲得するためのもの．
- ネットワーク側ではなく学習データ側にマスクをかけてあげることで双方向transformerが実現した．下図がモデルの概要．

<img src="https://cdn-images-1.medium.com/max/1600/1*ARMfhOTPxDWDiiAb-jFrvw.png">

- transformerモデルのEncoder部分を全結合的に接続したのがBERTモデル．

<img src="https://camo.qiitausercontent.com/0c71ab10a88d718eedb3ffadfc9a3c3339b9f7f2/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f3132333538392f63633966333864302d633934632d326334372d366466662d3235366239316166623431302e706e67" width=700>

- 上図のScaled Dot-Product Attentionはself-attention．attentionの重みを計算する際，softmaxで値が大きくなった時に勾配が0にならないようにsoftmaxのlogitのqueryとkeyの行列積を以下のように調整してあげる．

$$attenton \ weight = Softmax(\frac{q k^T}{\sqrt{depth}}) \\\  where \ depth = dim \ of \ embedding$$

<img src="https://camo.qiitausercontent.com/03b608cc2a33dd3a485eb440569560d4466b0e45/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f36313037392f61323533633837632d653631392d366431392d316233622d3632306430666535393936652e706e67">
### 事前学習タスクの選択
- どちらもBERTからはきだされた内部状態テンソルをInputとして一層のMLPでクラス分類しているだけ．

#### 事前学習1 マスク単語の予測
- 系列の15%を[MASK]トークンに置き換えて予測
- そのうち80%がマスク，10%がランダムな単語，10%を置き換えない方針で変換する

```python
class MaskedLanguageModel(nn.Module):
    """
    入力系列のMASKトークンから元の単語を予測する
    nクラス分類問題, nクラス : vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))
```

#### 事前学習2 隣接文の予測
- 二つの文章を与え隣り合っているかをYes/Noで判定
- 文章AとBが与えられた時に，50%の確率で別の文章Bに置き換える

```python
class NextSentencePrediction(nn.Module):
    """
    2クラス分類問題 : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))
```


### 実装

<img src="https://camo.qiitausercontent.com/348980102b722b9ab05ed175aa63f452af8ee1b0/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f3132333538392f66303532343866642d663737662d613164652d383336392d3036643363313735646139362e706e67" width=700>





## 汎用言語表現モデルのBioinformaticsへの応用
### 概要
- T細胞の抗原との結合度をバイナリで予測するモデルを構築したい．
- 使用するデータは，T細胞自身のペプチド配列と，T細胞が接触するタンパク質配列，またはMHC部分配列
- ペプチド配列が文字列的に扱えることから配列の解析に自然言語処理的アプローチを用いたい．

### アイデア
- BERTで試す前に一旦word2vecでモデルを組んでみるのはありかも

