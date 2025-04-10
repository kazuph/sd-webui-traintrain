import argparse
import os
import sys
import json
from pathlib import Path

# trainer モジュールをインポートできるようにパスを追加
project_root = Path(__file__).resolve().parent
# プロジェクトルートをパスに追加 (trainer パッケージをインポートするため)
if str(project_root) not in sys.path:
     sys.path.insert(0, str(project_root))
# trainer_path の追加は不要。プロジェクトルートから trainer パッケージを見つけるため。

# trainer モジュールのインポート
try:
    # trainer モジュール内の __init__.py で必要なものをインポートするようにする想定
    from trainer import trainer, dataset, lora # train を削除
    import trainer.train as train_module # train モジュールを別名でインポート
    # all_configs を trainer モジュールから取得
    all_configs = trainer.all_configs
except ImportError as e:
    print(f"Error importing trainer modules: {e}")
    print("Please ensure 'trainer' directory is in the Python path and has necessary __init__.py files.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)
except AttributeError as e:
     print(f"Error accessing trainer.all_configs: {e}")
     print("Please ensure 'trainer.py' defines 'all_configs'.")
     sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="TrainTrain CLI - LoRA Training Tool")

    # --- 必須引数 ---
    parser.add_argument("--mode", type=str, required=True, choices=["LoRA", "iLECO", "Difference", "ADDifT", "Multi-ADDifT"], help="Training mode.")
    # モデルパスのデフォルト値を環境変数から取得
    default_model = None
    if os.environ.get('STABLE_DIFFUSION_MODEL_FILENAME') and os.environ.get('STABLE_DIFFUSION_MODEL_DIR'):
        default_model = os.path.join(
            os.environ.get('STABLE_DIFFUSION_MODEL_DIR'),
            os.environ.get('STABLE_DIFFUSION_MODEL_FILENAME')
        )
    parser.add_argument("--model", type=str, required=False, default=default_model,
                       help="Path to the base model file (.safetensors or .ckpt). If not specified, uses STABLE_DIFFUSION_MODEL_DIR/STABLE_DIFFUSION_MODEL_FILENAME from environment.")
    parser.add_argument("--output_name", dest="save_lora_name", type=str, required=True, help="Filename for the output LoRA (without extension).") # Trainer uses save_lora_name

    # --- モード別必須引数 ---
    parser.add_argument("--data_dir", dest="lora_data_directory", type=str, help="Path to the directory containing training images (Required for LoRA, Multi-ADDifT modes).") # Trainer uses lora_data_directory
    parser.add_argument("--orig_prompt", type=str, help="Original prompt (Required for iLECO mode).")
    parser.add_argument("--targ_prompt", type=str, help="Target prompt (Required for iLECO mode).")
    parser.add_argument("--orig_image", type=str, help="Path to the original image (Required for Difference, ADDifT modes).")
    parser.add_argument("--targ_image", type=str, help="Path to the target image (Required for Difference, ADDifT modes).")

    # --- オプション引数 (JSON設定ファイルと共通) ---
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON configuration file.")
    parser.add_argument("--vae", type=str, default=None, help="Path to the VAE file (optional).")

    # --- オプション引数 (JSONを上書き可能) ---
    # all_configs から動的に引数を生成
    added_args = set(['mode', 'model', 'output_name', 'data_dir', 'orig_prompt', 'targ_prompt', 'orig_image', 'targ_image', 'config', 'vae'])
    for conf in all_configs:
        key = conf[0].split("(")[0] # Remove help text like (BASE=...)
        # Trainer内部名とCLI引数名をマッピング (必要に応じて追加)
        dest_map = {
            "lora_data_directory": "data_dir",
            "save_lora_name": "output_name",
            "train_learning_rate": "lr",
            "train_iterations": "iterations",
            "network_rank": "rank",
            "network_alpha": "alpha",
            # 他の引数も必要に応じて追加
        }
        cli_arg_name = dest_map.get(key, key)

        if cli_arg_name in added_args:
            continue # 既に定義済みの引数はスキップ

        arg_params = {
            "dest": key, # Trainer内部で使われるキー名を設定
            "type": conf[4],
            "help": f"{conf[0]} (Overrides value from JSON config)",
            "default": None # デフォルトはNoneにして、未指定時はJSONの値を使う
        }
        # Checkboxはstore_true/store_falseにする
        if conf[1] == "CH":
            arg_params["action"] = argparse.BooleanOptionalAction # --arg / --no-arg
            del arg_params["type"] # action='store_true'/'store_false' doesn't take type
        # Dropdownはchoicesを設定 (ただしCLIでは自由入力も許容する)
        elif conf[1] == "DD":
             arg_params["choices"] = conf[2] if conf[2] else None
             if arg_params["choices"] is None: # choicesが空ならtypeのみ
                 del arg_params["choices"]
             else: # choicesがある場合、typeはstrとして扱うことが多い
                 arg_params["type"] = str
        # CheckboxGroupはnargs='+'で複数選択可能にする
        elif conf[1] == "CB":
            arg_params["nargs"] = '+'
            arg_params["choices"] = conf[2] if conf[2] else None
            if arg_params["choices"] is None:
                 del arg_params["choices"]

        parser.add_argument(f"--{cli_arg_name.replace('_', '-')}", **arg_params)
        added_args.add(cli_arg_name)

    # save_overwrite は all_configs から動的に生成されるため、ここでの明示的な追加は不要


    args = parser.parse_args()

    # --- 引数の検証 ---
    # モデルが指定されていない場合に環境変数を確認
    if not args.model:
        if not (os.environ.get('STABLE_DIFFUSION_MODEL_FILENAME') and os.environ.get('STABLE_DIFFUSION_MODEL_DIR')):
            parser.error("--model is required or set both STABLE_DIFFUSION_MODEL_DIR and STABLE_DIFFUSION_MODEL_FILENAME environment variables.")
    
    if args.mode in ["LoRA", "Multi-ADDifT"] and not args.lora_data_directory and not args.config:
         parser.error("--data_dir or --config providing data_dir is required for LoRA and Multi-ADDifT modes.")
    if args.mode == "iLECO" and (not args.orig_prompt or not args.targ_prompt) and not args.config:
          parser.error("--orig_prompt and --targ_prompt or --config providing them are required for iLECO mode.")
    if args.mode in ["Difference", "ADDifT"] and (not args.orig_image or not args.targ_image) and not args.config:
          parser.error("--orig_image and --targ_image or --config providing them are required for Difference and ADDifT modes.")

    return args

def merge_configs(cli_args_dict, config_path):
    """Loads JSON config and merges CLI arguments, prioritizing CLI args."""
    config_data = {}
    if config_path:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            print(f"Loaded configuration from: {config_path}")
        except FileNotFoundError:
            print(f"Error: Configuration file not found at {config_path}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {config_path}")
            sys.exit(1)

    merged_config = config_data.copy()

    # CLI引数でJSON設定を上書き (Noneでない値のみ)
    for key, value in cli_args_dict.items():
        if value is not None:
            # 特殊なキー名のマッピング (argparseのdestとJSONキーが異なる場合)
            if key == "output_name": # argparseではoutput_nameだがTrainerはsave_lora_name
                 merged_config["save_lora_name"] = value
            elif key == "data_dir": # argparseではdata_dirだがTrainerはlora_data_directory
                 merged_config["lora_data_directory"] = value
            elif key == "lr":
                 merged_config["train_learning_rate"] = value
            elif key == "iterations":
                 merged_config["train_iterations"] = value
            elif key == "rank":
                 merged_config["network_rank"] = value
            elif key == "alpha":
                 merged_config["network_alpha"] = value
            # 他の引数も必要に応じてマッピング
            elif key in merged_config or key in [c[0].split("(")[0] for c in all_configs]: # all_configsのキー名もチェック
                 merged_config[key] = value
            else:
                 # print(f"Debug: CLI arg '{key}' not directly mapped or found in config, storing as is.")
                 merged_config[key] = value # マッピング外の引数も念のため保持

    # --- モデルパスが未設定の場合、環境変数を確認 ---
    if not merged_config.get('model') and os.environ.get('STABLE_DIFFUSION_MODEL_FILENAME') and os.environ.get('STABLE_DIFFUSION_MODEL_DIR'):
        merged_config['model'] = os.path.join(
            os.environ.get('STABLE_DIFFUSION_MODEL_DIR'),
            os.environ.get('STABLE_DIFFUSION_MODEL_FILENAME')
        )
        print(f"Using model from environment variables: {merged_config['model']}")

    # --- デフォルト値の設定 (JSONにもCLIにもない場合) ---
    # このブロックは Trainer.__init__ で all_configs を使ってデフォルト値を設定するため不要。削除する。
    # for conf in all_configs:
    #      key = conf[0].split("(")[0]
    #      if key not in merged_config:
    #           merged_config[key] = None
    # --- モード別必須パラメータの最終チェック ---
    # (Trainerクラス内でもチェックされるはずだが、早期にエラーを出す)
    mode = merged_config.get('mode')
    
    # モデルが指定されていない場合のエラーチェック
    if not merged_config.get('model'):
        print("Error: Model must be specified either via --model argument, JSON config, or environment variables.")
        sys.exit(1)
        
    if mode in ["LoRA", "Multi-ADDifT"] and not merged_config.get('lora_data_directory'):
        print(f"Error: --data_dir is required for mode '{mode}' (must be provided via CLI or config).")
        sys.exit(1)
    if mode == "iLECO" and (not merged_config.get('orig_prompt') or not merged_config.get('targ_prompt')):
        print(f"Error: --orig_prompt and --targ_prompt are required for mode '{mode}'.")
        sys.exit(1)
    if mode in ["Difference", "ADDifT"] and (not merged_config.get('orig_image') or not merged_config.get('targ_image')):
        print(f"Error: --orig_image and --targ_image are required for mode '{mode}'.")
        sys.exit(1)


    return merged_config

# create_trainer_args_list 関数は不要になったため削除 (関数定義全体を削除)

def main():
    args = parse_args()
    cli_args_dict = vars(args)

    merged_config = merge_configs(cli_args_dict, args.config)

    # --- Trainer の準備と実行 ---
    print("\n--- Starting Training ---")
    print("Effective Parameters:")
    # 主要なパラメータを表示（表示しすぎないように調整）
    print(f"  Mode: {merged_config.get('mode')}")
    # モデルパスとファイル名を表示
    model_path = merged_config.get('model', '')
    model_filename = os.path.basename(model_path) if model_path else 'None'
    print(f"  Model: {model_path}")
    print(f"  VAE: {merged_config.get('vae', 'Default')}")
    print(f"  Output Name: {merged_config.get('save_lora_name')}")
    if merged_config.get('lora_data_directory'):
        print(f"  Data Dir: {merged_config.get('lora_data_directory')}")
    if merged_config.get('orig_prompt'):
        print(f"  Orig Prompt: {merged_config.get('orig_prompt')}")
    if merged_config.get('targ_prompt'):
        print(f"  Targ Prompt: {merged_config.get('targ_prompt')}")
    if merged_config.get('orig_image'):
        print(f"  Orig Image: {merged_config.get('orig_image')}")
    if merged_config.get('targ_image'):
        print(f"  Targ Image: {merged_config.get('targ_image')}")
    print(f"  Rank: {merged_config.get('network_rank')}")
    print(f"  Alpha: {merged_config.get('network_alpha')}")
    print(f"  LR: {merged_config.get('train_learning_rate')}")
    print(f"  Iterations: {merged_config.get('train_iterations')}")
    print("--------------------------")

    try:
        # train_main を呼び出し
        # jsononly_or_paths は CLI では False 固定とする
        # 引数リストの代わりに merged_config ディクショナリを渡す
        result = train_module.train_main(
            False, # jsononly_or_paths
            merged_config # 設定ディクショナリを渡す
        )
        print("\n--- Training Result ---")
        print(result)
        print("-----------------------")

    except Exception as e:
        print(f"\n--- Training Failed ---")
        import traceback
        print(traceback.format_exc())
        print(f"Error: {e}")
        print("-----------------------")
        sys.exit(1)

if __name__ == "__main__":
    main()