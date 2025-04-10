FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/cache

WORKDIR /app

# システム依存関係のインストールと必要なパッケージをインストール（レイヤー削減のため一つにまとめる）
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && pip install --upgrade pip \
    && pip install numpy opencv-python-headless Pillow requests runpod pydantic \
    && pip install diffusers transformers accelerate safetensors torch xformers triton \
    && pip install onnxruntime>=1.21.0 \
    && mkdir -p /app/inputs /app/outputs /app/cache

RUN apt-get update && apt-get install -y curl tar build-essential
# COPY builder/download_e_withpost.sh /app/builder/download_e_withpost.sh
# RUN mkdir -p /app/models \
#     && cd /app && bash /app/builder/download_e_withpost.sh

# requirements.txt をコピーして依存関係をインストール
COPY requirements.txt ./
RUN pip install -r requirements.txt

# ソースコードのコピー
# アプリケーションコードのコピー ( .dockerignore で不要なファイルを除外することを推奨)
COPY . .

# Runpodワーカーのエントリーポイント（本番環境用）
CMD ["python", "cli.py"]
