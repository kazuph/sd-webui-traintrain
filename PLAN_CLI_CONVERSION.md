# sd-webui-traintrain CLI 化計画

## 1. 目標

`sd-webui-traintrain` を Stable Diffusion WebUI 拡張から独立したコマンドラインインターフェース (CLI) ツールに変換します。

## 2. 主要な変更点

*   Gradio UI コード (`scripts/traintrain.py`) を削除し、CLI のエントリーポイントとなる新しい Python スクリプト (例: `cli.py`) を作成します。
*   `trainer` ディレクトリ内の WebUI 依存コード (`standalone = False` のブロック) を削除または修正し、既存のスタンドアロンコードパス (`standalone = True`) に統合します。
*   Python 標準の `argparse` モジュールなどを利用して、コマンドライン引数を処理する仕組みを実装します。少なくとも以下の引数を受け取れるようにします。
    *   学習モード (`--mode`: LoRA, iLECO, Difference, ADDifT, Multi-ADDifT)
    *   モデルファイルのパス (`--model`)
    *   VAE ファイルのパス (`--vae`, オプション)
    *   設定 JSON ファイルのパス (`--config`)
    *   (LoRA/Multi-ADDifT モード用) データディレクトリ (`--data_dir`)
    *   (iLECO モード用) オリジナル/ターゲットプロンプト (`--orig_prompt`, `--targ_prompt`)
    *   (Difference/ADDifT モード用) オリジナル/ターゲット画像パス (`--orig_image`, `--targ_image`)
    *   出力 LoRA ファイル名 (`--output_name`)
    *   その他、JSON ファイルで指定しない場合の主要な学習パラメータ (ランク、アルファ、学習率、イテレーション数など)
*   学習の進捗状況 (例: tqdm のプログレスバー) や最終的な結果 (成功/失敗、出力ファイルパス) を標準出力 (コンソール) に表示するようにします。
*   プロジェクトの依存関係を見直し、WebUI 固有のライブラリ (例: Gradio の一部) を削除し、CLI 実行に必要なライブラリ (例: `torch`, `diffusers`, `accelerate`, `safetensors` など) を `requirements.txt` または `pyproject.toml` に明記します。

## 3. 維持する機能

*   既存の JSON ファイル (`presets/` や `jsons/` にあるもの) を `--config` 引数で指定して学習設定を読み込む機能。

## 4. 将来的な展望

*   CLI ツールとして安定動作を確認した後、必要に応じて FastAPI などを用いた Web API 化を検討します。

## 5. 計画の可視化

```mermaid
graph TD
    A[現状: WebUI拡張] --> B(目標: CLIツール化);
    B --> C[UIコード削除/CLIエントリーポイント作成];
    B --> D[WebUI依存コード削除/スタンドアロン化];
    B --> E[コマンドライン引数処理実装];
    E --> F{引数種別};
    F -- 主要パラメータ --> E;
    F -- JSON設定ファイルパス --> E;
    B --> G[標準出力への結果表示];
    B --> H[依存関係整理];
    C & D & E & G & H --> I(CLIツール完成);
    I --> J{将来: API化検討};