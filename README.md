# Chainer SDA MNIST Sample
Chainer (1.4系) で MNIST (手書き文字認識) データの次元圧縮を題材にして、Stacked Auto-Encoder を実行するサンプルです。  
  
当社技術ブログ「技ラボ」にて概要を公開しています。  
( http://wazalabo.com/chainer-stacked-auto-encoder.html )


## Requeirements
本コードは以下を使用しています。
- chainer
- numpy
- six
  
pip コマンドと requirements.txt から簡単にインストールを行うことが出来ます。

        $ sudo pip install -r requirements.txt


## データ
本ソースコードは MNIST データを題材としています。  
MNIST データローダは Chainer のサンプルソースコードを利用しています。  
( https://github.com/pfnet/chainer/blob/master/examples/mnist/data.py )


## 使用法
### 設定
設定は settings.py で行います。以下が設定可能です。
- GPU 使用可否
- 表示ログレベル
- 出力ファイル名
- 入力層、中間層のサイズ
- Pretraining 設定
- Finetuning 設定

### 実行
execute.py にて実行します。

        $ python execute.py




