# Fashion-CLIP

ファッション画像の分類を行うためのCLIPモデルを利用したPythonプロジェクトです。[patrickjohncyh/fashion-clip](https://huggingface.co/patrickjohncyh/fashion-clip)モデルを使用して、ファッション画像のスタイル分類を行います。

## 特徴

- HuggingFaceのCLIPモデルを使用したファッション画像の分類
- 複数のファッションカテゴリ（ストリートファッション、カジュアルスタイル、フォーマルウェア、スポーティスタイル、ビンテージスタイル）の判定
- 画像に対する各カテゴリの確率を出力

## 必要条件

- Python 3.10以上
- PyTorch
- transformers
- Pillow

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/akatsuki18/fashion-clip.git
cd fashion-clip

# 依存パッケージをインストール
pip install -r requirements.txt
```

## 使用方法

1. 分類したい画像を用意します
2. main.pyの画像パスを設定します
3. スクリプトを実行します

```bash
python main.py
```

## 出力例

```
a person wearing street fashion: 0.0004
a person wearing casual style: 0.3139
a person in formal clothes: 0.0327
a person wearing sporty outfit: 0.0740
a person with vintage style: 0.5789
```

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。