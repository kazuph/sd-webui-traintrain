import json
import os
import ast
import warnings
import torch
import subprocess
import sys
import torch.nn as nn
from datetime import datetime
from typing import Literal
from diffusers.optimization import get_scheduler
from transformers.optimization import AdafactorSchedule
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, CosineAnnealingWarmRestarts, StepLR, MultiStepLR, ReduceLROnPlateau, CyclicLR, OneCycleLR
from pprint import pprint
from accelerate import Accelerator
from pathlib import Path
import safetensors.torch
from diffusers.models import AutoencoderKL
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline, 
    FluxPipeline,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Standalone mode assumed for CLI conversion
path_root = Path(__file__).parent.parent # Assumes trainer.py is in trainer/ directory
lora_dir = os.path.join(path_root.parent,"output") # Default output dir relative to project root
path_trainer = os.path.join(path_root, "trainer")
# Note: Command line arguments will be handled by cli.py, replacing launch_utils.args

# --- Define constants used in all_configs first ---
SEP = "--------------------------"
OPTIMIZERS = ["AdamW", "AdamW8bit", "AdaFactor", "Lion", "Prodigy", SEP,
              "DadaptAdam","DadaptLion", "DAdaptAdaGrad", "DAdaptAdan", "DAdaptSGD",SEP,
               "Adam8bit", "SGDNesterov8bit", "Lion8bit", "PagedAdamW8bit", "PagedLion8bit",  SEP,
               "RAdamScheduleFree", "AdamWScheduleFree", "SGDScheduleFree", SEP,
               "CAME", "Tiger", "AdamMini",
               "PagedAdamW", "PagedAdamW32bit", "SGDNesterov", "Adam",]
POs = ["came", "tiger", "adammini"]
PASS2 = "2nd pass"
# --- End of constants definition ---
# --- all_configs definition from WebUI (scripts/traintrain.py) ---
# (注意: 実際のコードでは変数名やリストの内容が異なる可能性があるため、元のファイルで確認してください)
# 以下は構造を示すための例です。
NETWORK_TYPES = ["lierla", "c3lier","loha"] # 例
NETWORK_DIMS = [str(2**x) for x in range(11)] # 例
NETWORK_ALPHAS = [str(2**(x-5)) for x in range(16)] # 例
NETWORK_ELEMENTS = ["Full", "CrossAttention", "SelfAttention"] # 例
IMAGESTEPS = [str(x*64) for x in range(10)] # 例
LOSS_FUNCTIONS = ["MSE", "L1", "Smooth-L1"] # 例
SCHEDULERS = ["cosine_annealing", "cosine_annealing_with_restarts", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "piecewise_constant", "exponential", "step", "multi_step", "reduce_on_plateau", "cyclic", "one_cycle"] # 例
PRECISION_TYPES = ["fp32", "bf16", "fp16"] # CLI版に合わせて簡略化 (float32などを削除)
BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"] # 例
# 可視性フラグ (例) - CLIでは主にデフォルト値と型が重要
ALL = [True,True,True,True,True,True]
LORA = [True,False,False,False,False,False]
# ... (他の可視性フラグ) ...
DIFF2ND = [False,False,False,True,False,False] # use_2nd_pass_settings用

# パラメータ定義 (元の scripts/traintrain.py の定義をコピー)
network_type = ["network_type","DD",NETWORK_TYPES,NETWORK_TYPES[0],str,ALL]
network_rank = ["network_rank","DD",NETWORK_DIMS[2:],"16",int,ALL]
network_alpha = ["network_alpha","DD",NETWORK_ALPHAS,"8",float,ALL]
lora_data_directory = ["lora_data_directory","TX",None,"", str, LORA] # 可視性フラグは調整が必要かも
diff_target_name = ["diff_target_name","TX", None, "", str, LORA] # 可視性フラグは調整が必要かも
lora_trigger_word = ["lora_trigger_word","TX",None,"", str, LORA] # 可視性フラグは調整が必要かも
image_size = ["image_size(height, width)", "TX",None,"512,512",str,ALL] # デフォルト値を文字列に変更
train_iterations = ["train_iterations","TX",None,1000,int,ALL]
train_batch_size = ["train_batch_size", "TX",None,2,int,ALL]
train_learning_rate = ["train_learning_rate","TX",None,"1e-4",float,ALL]
train_optimizer =["train_optimizer","DD",OPTIMIZERS,OPTIMIZERS[0],str,ALL] # OPTIMIZERS は trainer.py で定義済み
train_optimizer_settings = ["train_optimizer_settings", "TX",None,"",str,ALL]
train_lr_scheduler =["train_lr_scheduler","DD",SCHEDULERS, "cosine",str,ALL]
train_lr_scheduler_settings = ["train_lr_scheduler_settings", "TX",None,"",str,ALL]
save_lora_name =  ["save_lora_name", "TX",None,"",str,ALL] # 可視性フラグは調整が必要かも
use_gradient_checkpointing = ["use_gradient_checkpointing","CH",None,False,bool,ALL]
network_blocks = ["network_blocks(BASE = TextEncoder)","CB",BLOCKID26,BLOCKID26,list,ALL]
# --- オプションパラメータも同様に追加 ---
network_conv_rank = ["network_conv_rank","DD",["0"] + NETWORK_DIMS[2:],"0",int,ALL]
network_conv_alpha = ["network_conv_alpha","DD",["0"] + NETWORK_ALPHAS,"0",float,ALL]
network_resume = ["network_resume","TX", None, "", str, LORA] # 可視性フラグは調整が必要かも
network_train_text_encoder =  ["network_train_text_encoder", "CH",None,False,bool,LORA] # 可視性フラグは調整が必要かも
network_element = ["network_element","DD",NETWORK_ELEMENTS,NETWORK_ELEMENTS[0],str,ALL] # デフォルト値を修正
network_strength  = ["network_strength","TX", None, 1.0, float, ALL] # float型に変更
train_loss_function =["train_loss_function","DD",LOSS_FUNCTIONS,"MSE",str,ALL]
train_seed = ["train_seed", "TX",None,-1,int, ALL]
train_min_timesteps = ["train_min_timesteps", "TX",None,0,int, ALL]
train_max_timesteps = ["train_max_timesteps", "TX",None,1000,int, ALL]
train_textencoder_learning_rate = ["train_textencoder_learning_rate","TX",None,"",float,LORA] # 可視性フラグは調整が必要かも
train_model_precision = ["train_model_precision","DD",PRECISION_TYPES,"fp16",str,ALL] # choices を PRECISION_TYPES に変更
train_lora_precision = ["train_lora_precision","DD",PRECISION_TYPES,"fp32",str,ALL] # choices を PRECISION_TYPES に変更
train_VAE_precision = ["train_VAE_precision","DD",PRECISION_TYPES,"fp32",str,ALL] # choices を PRECISION_TYPES に変更
image_buckets_step = ["image_buckets_step", "DD",IMAGESTEPS,"256",int,LORA] # 可視性フラグは調整が必要かも
image_num_multiply = ["image_num_multiply", "TX",None,1,int,LORA] # 可視性フラグは調整が必要かも
image_min_length = ["image_min_length", "TX",None,512,int,LORA] # 可視性フラグは調整が必要かも
image_max_ratio = ["image_max_ratio", "TX",None,2.0,float,LORA] # float型に変更, 可視性フラグは調整が必要かも
sub_image_num = ["sub_image_num", "TX",None,0,int,LORA] # 可視性フラグは調整が必要かも
image_mirroring =  ["image_mirroring", "CH",None,False,bool,LORA] # 可視性フラグは調整が必要かも
image_use_filename_as_tag =  ["image_use_filename_as_tag", "CH",None,False,bool,LORA] # 可視性フラグは調整が必要かも
image_disable_upscale = ["image_disable_upscale", "CH",None,False,bool,LORA] # 可視性フラグは調整が必要かも
image_use_transparent_background_ajust = ["image_use_transparent_background_ajust","CH",None,False,bool,ALL] # WebUI版から追加
save_per_steps = ["save_per_steps", "TX",None,0,int,ALL]
save_precision = ["save_precision","DD",PRECISION_TYPES,"fp16",str,ALL] # choices を PRECISION_TYPES に変更
save_overwrite = ["save_overwrite", "CH",None,False,bool,ALL]
save_as_json = ["save_as_json", "CH",None,False,bool,ALL] # 可視性フラグは調整が必要かも
diff_save_1st_pass = ["diff_save_1st_pass", "CH",None,False,bool,ALL] # 可視性フラグは調整が必要かも
diff_1st_pass_only = ["diff_1st_pass_only", "CH",None,False,bool,ALL] # 可視性フラグは調整が必要かも
diff_load_1st_pass = ["diff_load_1st_pass","TX", None, "", str, ALL] # 可視性フラグは調整が必要かも
diff_revert_original_target = ["diff_revert_original_target","CH", None, False, bool, ALL] # 可視性フラグは調整が必要かも
diff_use_diff_mask = ["diff_use_diff_mask","CH", None, False, bool, ALL] # 可視性フラグは調整が必要かも
diff_use_fixed_noise = ["diff_use_fixed_noise","CH", None, False, bool, ALL] # WebUI版から追加
diff_alt_ratio  = ["diff_alt_ratio","TX",None,"1.0",float,ALL] # float型に変更, デフォルト値を文字列に
train_lr_step_rules = ["train_lr_step_rules","TX",None,"",str,ALL]
train_lr_warmup_steps = ["train_lr_warmup_steps","TX",None,0,int,ALL]
train_lr_scheduler_num_cycles = ["train_lr_scheduler_num_cycles","TX",None,1,int,ALL]
train_lr_scheduler_power = ["train_lr_scheduler_power","TX",None, 1.0, float,ALL]
train_snr_gamma = ["train_snr_gamma","TX",None,5.0,float,ALL] # float型に変更
train_fixed_timsteps_in_batch = ["train_fixed_timsteps_in_batch","CH",None,False,bool,ALL]
logging_verbose = ["logging_verbose","CH",None,False,bool,ALL] # 可視性フラグは調整が必要かも
# logging_save_csv のデフォルト値を修正 (Checkbox なので boolean)
logging_save_csv = ["logging_save_csv","CH",None,False,bool,ALL] # デフォルトを False に
model_v_pred = ["model_v_pred", "CH",None,False,bool,ALL]
use_2nd_pass_settings = ["use_2nd_pass_settings", "CH", None, False, bool, DIFF2ND]

# 結合
all_configs = [
    # 必須に近いもの
    network_type, network_rank, network_alpha, lora_data_directory, diff_target_name, lora_trigger_word,
    image_size, train_iterations, train_batch_size, train_learning_rate,
    train_optimizer, train_optimizer_settings, train_lr_scheduler, train_lr_scheduler_settings, save_lora_name, use_gradient_checkpointing,
    network_blocks,
    # オプション
    network_resume, network_strength, network_conv_rank, network_conv_alpha, network_element, network_train_text_encoder,
    train_loss_function, train_seed, train_min_timesteps, train_max_timesteps, train_textencoder_learning_rate,
    train_model_precision, train_lora_precision, train_VAE_precision,
    image_buckets_step, image_num_multiply, image_min_length, image_max_ratio, sub_image_num, image_mirroring,
    image_use_filename_as_tag, image_disable_upscale, image_use_transparent_background_ajust,
    save_per_steps, save_precision, save_overwrite, save_as_json,
    diff_save_1st_pass, diff_1st_pass_only, diff_load_1st_pass, diff_revert_original_target, diff_use_diff_mask, diff_use_fixed_noise, diff_alt_ratio,
    train_lr_step_rules, train_lr_warmup_steps, train_lr_scheduler_num_cycles, train_lr_scheduler_power, train_snr_gamma, train_fixed_timsteps_in_batch,
    logging_verbose, logging_save_csv, model_v_pred,
    # 2nd pass 用フラグ
    use_2nd_pass_settings
]
# --- End of all_configs definition ---

jsonspath = os.path.join(path_root,"jsons")
logspath = os.path.join(path_root,"logs")
presetspath = os.path.join(path_root,"presets")

class Trainer():
    # values を config_dict に変更
    def __init__(self, jsononly, model, vae, mode, config_dict):
        if type(jsononly) is list:
            paths = jsononly
            jsononly = False
        else:
            paths = None # CLI モードではパスは config_dict 内にあるはず

        self.mode = mode
        self.use_8bit = False
        self.count_dict = {}
        self.metadata = {}

        # all_configs からデフォルト値を設定
        config_map = {conf[0]: conf for conf in all_configs}
        for conf in all_configs:
            key, _, _, default, dtype, _ = conf
            attr_key = key.split("(")[0] # "(...)" を除去
            # デフォルト値を正しい型に変換して設定
            try:
                if dtype == bool and isinstance(default, str):
                    default_value = default.lower() == 'true'
                elif dtype == list and isinstance(default, str): # リスト型の場合 (例: network_blocks)
                    # 文字列からリストに変換するロジックが必要な場合がある
                    # ここではデフォルトがリストであることを期待
                    default_value = default if isinstance(default, list) else []
                elif "precision" in attr_key:
                    # 精度関連のデフォルト値は文字列のまま保持し、parse_precisionで変換
                    default_value = parse_precision(default)
                else:
                    default_value = dtype(default)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert default value '{default}' for key '{attr_key}' to type {dtype}. Using raw default.")
                default_value = default # 変換失敗時はそのまま
            setattr(self, attr_key, default_value)

        # config_dict の内容で上書き & 型変換
        for key, value in config_dict.items():
            # config_dict のキーも "(...)" を除去して照合
            lookup_key = key.split("(")[0]
            conf = config_map.get(key) or config_map.get(lookup_key) # 元のキー or 除去後のキーで検索

            if conf:
                conf_key, _, _, _, dtype, _ = conf # all_configs から取得したキー名を使う
                attr_key = conf_key.split("(")[0] # 設定する属性名も all_configs のキーから取得

                try:
                    # 型変換ロジック
                    if dtype == bool:
                        if isinstance(value, str):
                            processed_value = value.lower() == 'true'
                        else:
                            processed_value = bool(value)
                    elif dtype == list:
                        # 文字列で渡された場合、カンマ区切りなどでリストに変換する処理が必要かも
                        if isinstance(value, str):
                             # 簡単なカンマ区切りリストのパース (必要に応じて調整)
                             processed_value = [item.strip() for item in value.split(',') if item.strip()]
                        elif isinstance(value, list):
                             processed_value = value
                        else:
                             print(f"Warning: Cannot convert value '{value}' for key '{attr_key}' to list. Using empty list.")
                             processed_value = [] # 不明な場合は空リスト
                    elif "precision" in attr_key:
                        if value is not None: # 値が None でない場合のみ parse_precision を呼ぶ
                             processed_value = parse_precision(value)
                             if attr_key == "train_model_precision" and value == "fp8":
                                 self.use_8bit = True
                                 print("Use 8bit Model Precision")
                        else:
                             # value が None の場合は、all_configs から設定されたデフォルト値 (parse_precision 適用済み) を使う
                             # getattr で現在の値を取得し、それが None ならデフォルトを使う（念のため）
                             current_value = getattr(self, attr_key, None)
                             if current_value is None:
                                  # all_configs からデフォルト値を取得し直して parse_precision を適用
                                  default_conf = config_map.get(conf_key)
                                  if default_conf:
                                       default_precision_str = default_conf[3]
                                       try:
                                            processed_value = parse_precision(default_precision_str)
                                            print(f"Warning: '{attr_key}' was None, using default parsed value: {processed_value}")
                                       except ValueError:
                                            print(f"Error: Could not parse default precision '{default_precision_str}' for '{attr_key}'.")
                                            processed_value = torch.float32 # フォールバック
                                  else:
                                       print(f"Error: Could not find default config for '{attr_key}'. Using fp32 as fallback.")
                                       processed_value = torch.float32 # フォールバック
                             else:
                                  processed_value = current_value # 既に設定されているデフォルト値を使う
                    elif attr_key == "train_optimizer_settings" or attr_key == "train_lr_scheduler_settings":
                        # Optimizer settings の処理 (setpass から移動)
                        dvalue = {}
                        if value is not None and isinstance(value, str) and len(value.strip()) > 0:
                            val_str = value.replace(" ", "").replace(";","\n")
                            args_list = val_str.split("\n")
                            for arg in args_list:
                                if "=" in arg:
                                    k, v_str = arg.split("=", 1)
                                    try:
                                        v = ast.literal_eval(v_str)
                                    except:
                                        v = v_str # 評価できない場合は文字列のまま
                                    dvalue[k.strip()] = v
                        processed_value = dvalue
                    elif attr_key == "train_optimizer" and isinstance(value, str):
                        processed_value = value.lower() # Optimizer 名を小文字に
                    else:
                        # 基本的な型変換
                        processed_value = dtype(value)

                    setattr(self, attr_key, processed_value)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not convert value '{value}' for key '{attr_key}' to type {dtype}. Using raw value. Error: {e}")
                    setattr(self, attr_key, value) # 変換失敗時は元の値を設定
            else:
                # all_configs にないキーはそのまま設定 (将来的な拡張用など)
                # この場合も attr_key は "(...)" を除去したものを使う
                attr_key_fallback = key.split("(")[0]
                setattr(self, attr_key_fallback, value)

        # --- 以前 __init__ や setpass で行っていた初期化や設定 ---
        self.save_dir = lora_dir # デフォルトの出力先
        if not os.path.exists(lora_dir):
            os.makedirs(lora_dir)

        # image_size の処理 (デフォルト値は all_configs で設定済み)
        # 文字列からリストへの変換とソート
        if isinstance(self.image_size, str):
             try:
                 self.image_size = [int(x.strip()) for x in self.image_size.split(",")]
             except ValueError:
                 print(f"Warning: Invalid image_size format '{self.image_size}'. Using default [512, 512].")
                 # all_configs からデフォルト値を取得し直す
                 img_size_conf = config_map.get("image_size(height, width)")
                 default_img_size_str = img_size_conf[3] if img_size_conf else "512,512"
                 try:
                     self.image_size = [int(x.strip()) for x in default_img_size_str.split(",")]
                 except ValueError:
                     self.image_size = [512, 512] # 再度フォールバック
        if len(self.image_size) == 1:
             self.image_size = self.image_size * 2
        self.image_size.sort()

        # その他の初期化 (デフォルト値は all_configs で設定済み)
        self.total_images = 0
        # save_1st_pass は diff_1st_pass_only によって上書きされる可能性がある
        if getattr(self, 'diff_1st_pass_only', False):
             self.save_1st_pass = True # diff_1st_pass_only が True なら save_1st_pass も True に
        else:
             # diff_1st_pass_only が False の場合、diff_save_1st_pass の値を使う
             # diff_save_1st_pass は all_configs でデフォルト値が設定されているはず
             self.save_1st_pass = getattr(self, 'diff_save_1st_pass', False)

        # gradient_accumulation_steps と train_repeat は all_configs にないので getattr で取得
        # TODO: all_configs に gradient_accumulation_steps と train_repeat を追加検討
        self.gradient_accumulation_steps = getattr(self, 'gradient_accumulation_steps', 1)
        self.train_repeat = getattr(self, 'train_repeat', 1)

        # プロンプトと画像の取得 (config_dict から)
        self.prompts = [
            config_dict.get('orig_prompt', ''),
            config_dict.get('targ_prompt', ''),
            config_dict.get('neg_prompt', '') # neg_prompt も取得試行
        ]
        self.images = [
            config_dict.get('orig_image', None),
            config_dict.get('targ_image', None)
        ]

        # add_dcit の作成
        self.add_dcit = {
            "mode": self.mode,
            "model": model, # __init__ に渡された model を使用
            "vae": vae,     # __init__ に渡された vae を使用
            "original prompt": self.prompts[0],
            "target prompt": self.prompts[1]
        }

        # 以前 setpass 後に呼び出していたメソッド
        self.mode_fixer()
        self.checkfile() # save_lora_name が設定された後に呼び出す必要あり

        # JSON エクスポートとパス設定
        self.export_json(jsononly)
        if paths is not None:
            self.setpaths(paths) # paths があれば設定

    # setpass メソッドは __init__ に統合されたため削除

    savedata = ["model", "vae", ]
    
    def export_json(self, jsononly):
        current_time = datetime.now()
        # export_json で config_dict の内容を使うように変更 (setpass 呼び出しを削除)
        # outdict を生成する際に dtype を文字列に変換
        outdict = {}
        exclude_keys = {'a', 'unet', 'text_model', 'vae', 'noise_scheduler', 'dataloader', 'orig_cond', 'targ_cond', 'un_cond', 'orig_vector', 'targ_vector', 'un_vector', 'orig_latent', 'targ_latent'} # JSONに含めない属性
        for k, v in self.__dict__.items():
             if not k.startswith('_') and not callable(v) and k not in exclude_keys:
                 if isinstance(v, torch.dtype):
                     # dtype を文字列に変換
                     if v == torch.float32:
                         outdict[k] = "fp32"
                     elif v == torch.float16:
                         outdict[k] = "fp16"
                     elif v == torch.bfloat16:
                         outdict[k] = "bf16"
                     else:
                         outdict[k] = str(v) # 不明な dtype は文字列化
                 elif isinstance(v, torch.device):
                      outdict[k] = str(v) # device オブジェクトも文字列化
                 # 他にもシリアライズできない型があればここに追加
                 else:
                      # リスト内の dtype もチェック (例: network_blocks) - より堅牢なチェックが必要な場合あり
                      if isinstance(v, list):
                           try:
                               # リストの内容がJSONシリアライズ可能か簡易チェック
                               json.dumps(v)
                               outdict[k] = v
                           except TypeError:
                                # シリアライズできないリストはスキップするか、代替表現にする
                                print(f"Warning: Skipping non-serializable list attribute '{k}' in export_json.")
                                pass # または outdict[k] = str(v) など
                      else:
                           outdict[k] = v
        # TODO: Difference モードの 2nd pass 設定のエクスポートロジックを再実装する必要がある
        # if self.mode == "Difference" and getattr(self, 'use_2nd_pass_settings', False):
        #     pass # 2nd pass の設定を outdict に追加する処理
        outdict.update(self.add_dcit)
        today = current_time.strftime("%Y%m%d")
        time = current_time.strftime("%Y%m%d_%H%M%S")
        add = "" if jsononly else f"-{time}"
        jsonpath = os.path.join(presetspath, self.save_lora_name + add + ".json")  if jsononly else os.path.join(jsonspath, today, self.save_lora_name + add + ".json")
        self.csvpath = os.path.join(logspath ,today, self.save_lora_name + add + ".csv")
        
        if self.save_as_json:
            directory = os.path.dirname(jsonpath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(jsonpath, 'w') as file:
                json.dump(outdict, file, indent=4)
        
        if jsononly:
            with open(jsonpath, 'w') as file:
                json.dump(outdict, file, indent=4)  

    def db(self, *obj, pp = False):
        if self.logging_verbose:
            if pp:
                pprint(*obj)
            else:
                print(*obj)

    def checkfile(self):
        if self.save_lora_name == "":
            self.save_lora_name = "untitled"

        filename = os.path.join(self.save_dir, f"{self.save_lora_name}.safetensors")

        self.isfile = os.path.isfile(filename) and not self.save_overwrite
    
    def tagcount(self, prompt):
        tags = [p.strip() for p in prompt.split(",")]

        for tag in tags:
            if tag in self.count_dict:
                self.count_dict[tag] += 1
            else:
                self.count_dict[tag] = 1
    
    #["LoRA", "iLECO", "Difference","ADDifT", "Multi-ADDifT"]
    def mode_fixer(self):
        if self.mode == "LoRA":
            pass

        if self.mode == "iLECO":
            self.network_resume  = ""

        if self.mode == "ADDifT":
            if self.diff_load_1st_pass:
                self.network_resume = self.diff_load_1st_pass
            if self.lora_trigger_word == "" and "BASE" in self.network_blocks:
                self.network_blocks.remove("BASE")

        if self.mode == "Multi-ADDifT":
            if self.diff_load_1st_pass:
                self.network_resume = self.diff_load_1st_pass

    def sd_typer(self, ver = None):
        # WebUI specific model detection removed. Version must be passed via 'ver'.
        if ver is None:
             raise ValueError("Model version ('ver') must be provided in CLI mode.")
        else:
            self.is_sd1 = ver == 0
            self.is_sd2 = ver == 1
            self.is_sdxl = ver == 2
            self.is_sd3 = ver == 3
            self.is_flux = ver == 4

        #sdver: 0:sd1 ,1:sd2, 2:sdxl, 3:sd3, 4:flux
        if self.is_sdxl:
            self.model_version = "sdxl_base_v1-0"
            self.vae_scale_factor = 0.13025
            self.vae_shift_factor = 0
            self.sdver = 2
        elif self.is_sd2:
            self.model_version = "sd_v2"
            self.vae_scale_factor = 0.18215
            self.vae_shift_factor = 0
            self.sdver = 1
        elif self.is_sd3:
            self.model_version = "sd_v3"
            self.vae_scale_factor = 1.5305
            self.vae_shift_factor = 0.0609
            self.sdver = 3
        elif self.is_flux:
            self.model_version = "flux"
            self.vae_scale_factor = 0.3611
            self.vae_shift_factor = 0.1159
            self.sdver = 4
        else:
            self.model_version = "sd_v1"
            self.vae_scale_factor = 0.18215
            self.vae_shift_factor = 0
            self.sdver = 0
        
        self.is_dit = self.is_sd3 or self.is_flux
        self.is_te2 = self.is_sdxl or self.is_sd3

        print("Base Model : ", self.model_version)
    
    def setpaths(self, paths):
        if paths[0] is not None:#root
            self.save_dir = os.path.join(paths[0],"lora")
            self.models_dir = os.path.join(paths[0],"StableDiffusion")
            self.vae_dir = os.path.join(paths[0],"VAE")
        if paths[1] is not None:#model
            self.models_dir = paths[1]
        if paths[2] is not None:#vae
            self.vae_dir = paths[2]
        if paths[3] is not None:#lora
            self.save_dir = paths[3]

def import_json(name, preset = False, cli = False):
    def find_files(file_name):
        for root, dirs, files in os.walk(jsonspath):
            if file_name in files:
                return os.path.join(root, file_name)
        return None
    if preset:
        filepath = os.path.join(presetspath, name + ".json")
    elif cli:
        filepath = name
    else:
        filepath = find_files(name if ".json" in name else name + ".json")

    output = []

    if filepath is None:
        # Return None or raise error in CLI mode if file not found
        # Returning empty list for now, cli.py should handle the error
        print(f"Warning: Config file '{name}' not found.")
        return None
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # all_configs を参照して設定を読み込むヘルパー関数
    def setconfigs_from_json(json_data, config_list):
        output_dict = {}
        config_map = {conf[0]: conf for conf in config_list} # キーで検索できるようにマップ作成

        for conf in config_list:
            key, _, _, default, dtype, _ = conf
            attr_key = key.split("(")[0] # "(...)" を除去
            value = json_data.get(key) # JSONから値を取得

            if value is not None:
                # JSONに値が存在する場合、型変換を試みる
                try:
                    if dtype == bool:
                        if isinstance(value, str):
                            processed_value = value.lower() == 'true'
                        else:
                            processed_value = bool(value)
                    elif dtype == list:
                         if isinstance(value, str):
                             processed_value = [item.strip() for item in value.split(',') if item.strip()]
                         elif isinstance(value, list):
                             processed_value = value
                         else:
                             processed_value = default # 不明な場合はデフォルト
                    elif "precision" in attr_key:
                        # 精度関連は文字列のまま保持 (Trainer.__init__でtorch.dtypeに変換)
                        processed_value = str(value)
                    elif attr_key == "train_optimizer_settings" or attr_key == "train_lr_scheduler_settings":
                        # 文字列で保存されているはずなので、そのまま保持 (Trainer.__init__で辞書に変換)
                        processed_value = value if isinstance(value, str) else ""
                    elif attr_key == "train_optimizer":
                         # OPTIMIZERS リストと比較して正式名称に変換
                         found_optim = default # デフォルト値で初期化
                         if isinstance(value, str):
                              for optim in OPTIMIZERS:
                                   if value.lower() == optim.lower():
                                        found_optim = optim
                                        break
                         processed_value = found_optim
                    else:
                        processed_value = dtype(value) # 基本的な型変換
                    output_dict[key] = processed_value
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert JSON value '{value}' for key '{key}' to type {dtype}. Using default.")
                    # デフォルト値を正しい型に変換して設定
                    try:
                        if dtype == bool and isinstance(default, str):
                            default_value = default.lower() == 'true'
                        elif dtype == list and isinstance(default, str):
                            default_value = default if isinstance(default, list) else []
                        elif "precision" in attr_key:
                            default_value = str(default) # 文字列として保持
                        else:
                            default_value = dtype(default)
                    except (ValueError, TypeError):
                         default_value = default # 再度失敗したらそのまま
                    output_dict[key] = default_value
            else:
                # JSONに値が存在しない場合はデフォルト値を使用 (正しい型で)
                try:
                    if dtype == bool and isinstance(default, str):
                        default_value = default.lower() == 'true'
                    elif dtype == list and isinstance(default, str):
                        default_value = default if isinstance(default, list) else []
                    elif "precision" in attr_key:
                        default_value = str(default) # 文字列として保持
                    else:
                        default_value = dtype(default)
                except (ValueError, TypeError):
                     default_value = default # 再度失敗したらそのまま
                output_dict[key] = default_value
        return output_dict

    # JSONデータから設定を読み込む
    config_data = setconfigs_from_json(data, all_configs)

    # 2nd pass の設定があれば上書き
    if PASS2 in data and data[PASS2] and isinstance(data[PASS2], dict):
        second_pass_data = setconfigs_from_json(data[PASS2], all_configs)
        # 2nd pass 用のキーを生成してマージ (例: "train_iterations_2nd_pass")
        # この部分はWebUI版の挙動に合わせる必要あり。現状は単純上書き。
        # TODO: 2nd pass のキー命名規則を確認し、適切にマージする
        # config_data.update(second_pass_data) # 単純上書きは問題を起こす可能性があるのでコメントアウト
        print("Warning: 2nd pass settings found in JSON but merging logic is not fully implemented yet.")


    # プロンプトと画像パスを追加
    config_data["original prompt"] = data.get("original prompt", "")
    config_data["target prompt"] = data.get("target prompt", "")
    config_data["neg_prompt"] = data.get("neg_prompt", "") # neg_prompt も追加
    if cli:
        config_data["original image"] = data.get("original image", "")
        config_data["target image"] = data.get("target image", "")

    # mode, model, vae を追加
    config_data["mode"] = data.get("mode", "LoRA")
    config_data["model"] = data.get("model", None)
    config_data["vae"] = data.get("vae", None)

    return config_data # 辞書形式で返す

    if PASS2 in data and data[PASS2]:
        setconfigs(data[PASS2], output)
    else:
        output = output * 2

    output.append(data["original prompt"] if "original prompt" in data else "")
    output.append(data["target prompt"] if "target prompt" in data else "")
    output.append("")
    if cli:
        output.append(data["original image"] if "original image" in data else "")
        output.append(data["target image"] if "target image" in data else "")
    
    head = []
    head.append(data["mode"] if "mode" in data else "LoRA")
    head.append(data["model"] if "model" in data else None)
    head.append(data["vae"] if "vae" in data else None)

    return head + output

AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]

def get_optimizer(name: str, trainable_params, lr, optimizer_kwargs, network):
    name = name.lower()
    if name.startswith("dadapt"):
        import dadaptation
        if name == "dadaptadam":
            optim = dadaptation.DAdaptAdam
        elif name == "dadaptlion":
            optim = dadaptation.DAdaptLion               
        elif name == "DAdaptAdaGrad".lower():
            optim = dadaptation.DAdaptAdaGrad
        elif name == "DAdaptAdan".lower():
            optim = dadaptation.DAdaptAdan
        elif name == "DAdaptSGD".lower():
            optim = dadaptation.DAdaptSGD

    elif name.endswith("8bit"): 
        import bitsandbytes as bnb
        try:
            if name == "adam8bit":
                optim = bnb.optim.Adam8bit
            elif name == "adamw8bit":  
                optim = bnb.optim.AdamW8bit
            elif name == "SGDNesterov8bit".lower():
                optim = bnb.optim.SGD8bit
                if "momentum" not in optimizer_kwargs:
                    optimizer_kwargs["momentum"] = 0.9
                optimizer_kwargs["nesterov"] = True
            elif name == "Lion8bit".lower():
                optim = bnb.optim.Lion8bit
            elif name == "PagedAdamW8bit".lower():
                optim = bnb.optim.PagedAdamW8bit
            elif name == "PagedLion8bit".lower():
                optim  = bnb.optim.PagedLion8bit

        except AttributeError:
            raise AttributeError(
                f"No {name}. The version of bitsandbytes installed seems to be old. Please install newest. / {name}が見つかりません。インストールされているbitsandbytesのバージョンが古いようです。最新版をインストールしてください。"
            )

    elif name.lower() == "adafactor":
        import transformers
        optim = transformers.optimization.Adafactor

    elif name == "PagedAdamW".lower():
        import bitsandbytes as bnb
        optim = bnb.optim.PagedAdamW
    elif name == "PagedAdamW32bit".lower():
        import bitsandbytes as bnb
        optim = bnb.optim.PagedAdamW32bit

    elif name == "SGDNesterov".lower():
        if "momentum" not in optimizer_kwargs:
            optimizer_kwargs["momentum"] = 0.9
        optimizer_kwargs["nesterov"] = True
        optim = torch.optim.SGD

    elif name.endswith("schedulefree".lower()):
        import schedulefree as sf
        if name == "RAdamScheduleFree".lower():
            optim = sf.RAdamScheduleFree
        elif name == "AdamWScheduleFree".lower():
            optim = sf.AdamWScheduleFree
        elif name == "SGDScheduleFree".lower():
            optim = sf.SGDScheduleFree

    elif name in POs:
        import pytorch_optimizer as po    
        if name == "CAME".lower():
            optim = po.CAME        
        elif name == "Tiger".lower():
            optim = po.Tiger        
        elif name == "AdamMini".lower():
            optim = po.AdamMini   
        
    else:
        if name == "adam":
            optim = torch.optim.Adam
        elif name == "adamw":
            optim = torch.optim.AdamW  
        elif name == "lion":
            from lion_pytorch import Lion
            optim = Lion
        elif name == "prodigy":
            import prodigyopt
            optim = prodigyopt.Prodigy

    
    if name.startswith("DAdapt".lower()) or name == "Prodigy".lower():
    # check lr and lr_count, and logger.info warning
        actual_lr = lr
        lr_count = 1
        if type(trainable_params) == list and type(trainable_params[0]) == dict:
            lrs = set()
            actual_lr = trainable_params[0].get("lr", actual_lr)
            for group in trainable_params:
                lrs.add(group.get("lr", actual_lr))
            lr_count = len(lrs)

        if actual_lr <= 0.1:
            print(
                f"learning rate is too low. If using D-Adaptation or Prodigy, set learning rate around 1.0 / 学習率が低すぎるようです。D-AdaptationまたはProdigyの使用時は1.0前後の値を指定してください: lr={actual_lr}"
            )
            print("recommend option: lr=1.0 / 推奨は1.0です")
        if lr_count > 1:
            print(
                f"when multiple learning rates are specified with dadaptation (e.g. for Text Encoder and U-Net), only the first one will take effect / D-AdaptationまたはProdigyで複数の学習率を指定した場合（Text EncoderとU-Netなど）、最初の学習率のみが有効になります: lr={actual_lr}"
            )

    elif name == "Adafactor".lower():
        # 引数を確認して適宜補正する
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True  # default
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
            print(
                f"set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします"
            )
            optimizer_kwargs["relative_step"] = True
 
        if optimizer_kwargs["relative_step"]:
            print(f"relative_step is true / relative_stepがtrueです")
            if lr != 0.0:
                print(f"learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
  

            # trainable_paramsがgroupだった時の処理：lrを削除する
            if type(trainable_params) == list and type(trainable_params[0]) == dict:
                has_group_lr = False
                for group in trainable_params:
                    p = group.pop("lr", None)
                    has_group_lr = has_group_lr or (p is not None)

            lr = None
        #TODO

        # else:
        #     if args.max_grad_norm != 0.0:
        #         logger.warning(
        #             f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません"
        #         )
        #     if args.lr_scheduler != "constant_with_warmup":
        #         logger.warning(f"constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
        #     if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
        #         logger.warning(f"clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")


    return optim(network, lr = lr, **optimizer_kwargs) if name == "AdamMini".lower() else  optim(trainable_params, lr = lr, **optimizer_kwargs) 


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2

    step = 64

    min_step = min_resolution // step
    max_step = max_resolution // step

    height = torch.randint(min_step, max_step, (1,)).item() * step
    width = torch.randint(min_step, max_step, (1,)).item() * step

    return height, width

def load_noise_scheduler(name: str,v_parameterization: bool):
    sched_init_args = {}
    name = name.lower().replace(" ", "_")
    if name == "ddim":
        scheduler_cls = DDIMScheduler
    elif name == "ddpm":
        scheduler_cls = DDPMScheduler
    elif name == "pndm":
        scheduler_cls = PNDMScheduler
    elif name == "lms" or name == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif name == "euler" or name == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif name == "euler_a" or name == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif name == "dpmsolver" or name == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = name
    elif name == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif name == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif name == "dpm_2" or name == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif name == "dpm_2_a" or name == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler
        "Selected scheduler is not in list, use DDIMScheduler."

    if v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    scheduler = scheduler_cls(
        num_train_timesteps = 1000,
        beta_start = 0.00085,
        beta_end = 0.0120, 
        beta_schedule = "scaled_linear",
        **sched_init_args,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # print("set clip_sample to True")
        scheduler.config.clip_sample = True

    prepare_scheduler_for_custom_training(scheduler)

    return scheduler

def prepare_scheduler_for_custom_training(noise_scheduler):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to("cuda")

def load_checkpoint_model(checkpoint_path, t, clip_skip = None, vae = None):
    pipe = StableDiffusionPipeline.from_single_file(checkpoint_path,upcast_attention=True if t.is_sd2 else False)

    text_encoder = pipe.text_encoder
    unet = pipe.unet
    if vae is None and hasattr(pipe, "vae"):
        vae = pipe.vae

    if clip_skip is not None:
        if t.is_sd2:
            text_encoder.config.num_hidden_layers = 24 - (clip_skip - 1)
        else:
            text_encoder.config.num_hidden_layers = 12 - (clip_skip - 1)

    text_model = TextModel(
        pipe.tokenizer,
        pipe.text_encoder,
        None, None, None, None,
        t.sdver,
        clip_skip
    )
    
    del pipe
    return text_model, unet, vae

def load_checkpoint_model_xl(checkpoint_path, t, vae = None):
    pipe = StableDiffusionXLPipeline.from_single_file(checkpoint_path)

    unet = pipe.unet
    if vae is None and hasattr(pipe, "vae"):
        vae = pipe.vae

    text_model = TextModel(
        pipe.tokenizer,
        pipe.text_encoder,
        pipe.tokenizer_2,
        pipe.text_encoder_2,
        None, None,
        t.sdver,
    )
    del pipe
    return text_model, unet, vae

def load_checkpoint_model_sd3(checkpoint_path, t, vae = None, clip_l = None, clip_g = None, t5 = None):
    pipe = StableDiffusion3Pipeline.from_single_file(checkpoint_path)

    unet = pipe.transformer
    vae = pipe.vae

    assert clip_l is None and pipe.text_encoder, "ERROR, No CLIP L"
    assert clip_g is None and pipe.text_encoder2, "ERROR, No CLIP g"
    assert t5 is None and pipe.text_encoder3, "ERROR, No T5"

    text_model = TextModel(
        pipe.tokenizer,
        pipe.text_encoder, 
        pipe.tokenizer_2,
        pipe.text_encoder_2,
        pipe.tokenizer_3,
        pipe.text_encoder_3,
        t.sdver
    )
    
    del pipe
    return text_model, unet, vae

def load_checkpoint_model_flux(checkpoint_path, t, vae = None, clip_l = None, clip_g = None, t5 = None):
    pipe = FluxPipeline.from_single_file(checkpoint_path)

    unet = pipe.transformer
    vae = pipe.vae

    assert clip_l is None and pipe.text_encoder, "ERROR, No CLIP L"
    assert t5 is None and pipe.text_encoder3, "ERROR, No T5"

    text_model = TextModel(
        pipe.tokenizer,
        pipe.text_encoder, 
        None, None,
        pipe.tokenizer_2,
        pipe.text_encoder_2,
        t.sdver
    )
    
    del pipe
    return text_model, unet, vae

class TextModel(nn.Module):
    def __init__(self, cl_tn, cl_en, cg_tn, cg_en, t5_tn, t5_en, sdver, clip_skip=-1):
        super().__init__()
        self.tokenizers = [cl_tn, cg_tn, t5_tn]
        self.text_encoders = nn.ModuleList([cl_en, cg_en, t5_en])
        self.clip_skip = clip_skip if clip_skip is not None else -1
        self.textual_inversion = False
        self.sdver = sdver

    def tokenize(self, texts):
        tokens = []
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            if tokenizer is None:
                tokens.append(None)
                continue
            token = tokenizer(
                texts, 
                max_length=tokenizer.model_max_length, 
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            ).input_ids.to(text_encoder.device)
            tokens.append(token)
        return tokens

    #sdver: 0:sd1 ,1:sd2, 2:sdxl, 3:sd3, 4:flux
    def forward(self, tokens):
        if 2 > self.sdver:
            return self.encode_sd1_2(tokens)
        elif self.sdver == 2:
            return self.encode_sdxl(tokens)
        elif self.sdver == 3:
            return self.encode_sd3(tokens)
        elif self.sdver == 4:
            return self.encode_flux(tokens)

    def encode_sd1_2(self, tokens):
        encoder_hidden_states = self.text_encoders[0](tokens[0], output_hidden_states=True).hidden_states[self.clip_skip]
        encoder_hidden_states = self.text_encoders[0].text_model.final_layer_norm(encoder_hidden_states)
        return encoder_hidden_states, None
    
    def encode_sdxl(self, tokens):
        encoder_hidden_states = self.text_encoders[0](tokens[0], output_hidden_states=True).hidden_states[self.clip_skip]
        encoder_output_2 = self.text_encoders[1](tokens[1], output_hidden_states=True)
        last_hidden_state = encoder_output_2.last_hidden_state

        # calculate pooled_output
        eos_token_index = torch.where(tokens[1] == self.tokenizers[1].eos_token_id)[1].to(device=last_hidden_state.device)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),eos_token_index]
        pooled_output = self.text_encoders[1].text_projection(pooled_output)

        encoder_hidden_states_2 = encoder_output_2.hidden_states[self.clip_skip]

        # (b, n, 768) + (b, n, 1280) -> (b, n, 2048)
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=2)

        # pooled_output is zero vector for empty text            
        for i, token in enumerate(tokens[1]):
            if token[1].item() == self.tokenizers[1].eos_token_id:
                pooled_output[i] = 0

        return encoder_hidden_states, pooled_output
    
    def encode_sd3(self, tokens):
        encoder_output = self.text_encoders[0](tokens[0], output_hidden_states=True)
        last_hidden_state = encoder_output.last_hidden_state
        # calculate pooled_output
        eos_token_index = torch.where(tokens[0] == self.tokenizers[0].eos_token_id)[1][0].to(device=last_hidden_state.device)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index]
        pooled_output = self.text_encoders[0].text_projection(pooled_output)

        encoder_hidden_states = encoder_output.hidden_states[self.clip_skip]

        encoder_output_2 = self.text_encoders[1](tokens[1], output_hidden_states=True)
        last_hidden_state = encoder_output_2.last_hidden_state
        # calculate pooled_output
        eos_token_index = torch.where(tokens[1] == self.tokenizers[1].eos_token_id)[1].to(device=last_hidden_state.device)
        pooled_output_2 = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),eos_token_index]
        pooled_output_2 = self.text_encoders[1].text_projection(pooled_output_2)

        encoder_hidden_states_2 = encoder_output_2.hidden_states[self.clip_skip]

        encoder_hidden_states_3 = self.text_encoders[2](tokens[2], output_hidden_states=False)[0]

        # (b, n, 768) + (b, n, 1280) -> (b, n, 2048)
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=2)

        # pad
        encoder_hidden_states = torch.cat([encoder_hidden_states, torch.zeros_like(encoder_hidden_states)], dim=2)
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_3], dim=1) # t5

        pooled_output = torch.cat([pooled_output, pooled_output_2], dim=1)

        return encoder_hidden_states, pooled_output

    def encode_flux(self, tokens):
        pooled_output = self.text_encoders[0](tokens[0], output_hidden_states=False).pooler_output
        encoder_hidden_states = self.text_encoders[2](tokens[2], output_hidden_states=False)[0]

        return encoder_hidden_states, pooled_output

    def encode_text(self, texts):
        tokens = self.tokenize(texts)
        encoder_hidden_states, pooled_output = self.forward(tokens)
        return encoder_hidden_states, pooled_output

    def gradient_checkpointing_enable(self, enable=True):
        if enable:
            for te in self.text_encoders:
                if te is not None:
                    for param in te.parameters():
                        param.requires_grad = True
                    te.gradient_checkpointing_enable()
        else:
            for te in self.text_encoders:
                if te is not None:
                    te.gradient_checkpointing_disable()

    def to(self, device = None, dtype = None):
        for te in self.text_encoders:
            if te is not None:
                te = te.to(device,dtype)

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def make_accelerator(t):
    accelerator = Accelerator(
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        mixed_precision=parse_precision(t.train_model_precision, mode = False)
    )

    return accelerator

def parse_precision(precision, mode = True):
    if mode:
        if precision == "fp32" or precision == "float32":
            return torch.float32
        elif precision == "fp16" or precision == "float16" or precision == "fp8":
            return torch.float16
        elif precision == "bf16" or precision == "bfloat16":
            return torch.bfloat16
    else:
        if precision == torch.float16 or precision == "fp8":
            return 'fp16'
        elif precision == torch.bfloat16:
            return 'bf16'

    raise ValueError(f"Invalid precision type: {precision}")
    
def load_lr_scheduler(t, optimizer):
    if t.train_optimizer == "adafactor":
        return AdafactorSchedule(optimizer)
    
    args = t.train_lr_scheduler_settings
    print(f"LR Scheduler args: {args}")
    
    # アニーリング系のスケジューラを追加
    if t.train_lr_scheduler == "cosine_annealing":
        print("Using CosineAnnealingLR")
        return CosineAnnealingLR(
            optimizer,
            T_max=t.train_iterations,
            **args
        )
    elif t.train_lr_scheduler == "cosine_annealing_with_restarts":
        print("Using CosineAnnealingWarmRestarts")
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.pop("T_0") if "T_0" in args else 10,
            **args
        )
    elif t.train_lr_scheduler == "exponential":
        print("Using ExponentialLR")
        return ExponentialLR(
            optimizer,
            gamma=args.pop("gamma") if "gamma" in args else 0.9,
            **args
        )
    elif t.train_lr_scheduler == "step":
        print("Using StepLR")
        return StepLR(
            optimizer,
            step_size=args.pop("step_size") if "step_size" in args else 10,
            **args
        )
    elif t.train_lr_scheduler == "multi_step":
        print("Using MultiStepLR")
        return MultiStepLR(
            optimizer,
            milestones=args.pop("milestones") if "milestones" in args else [30, 60, 90],
            **args
        )
    elif t.train_lr_scheduler == "reduce_on_plateau":
        print("Using ReduceLROnPlateau")
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            **args
        )
    elif t.train_lr_scheduler == "cyclic":
        print("Using CyclicLR")
        return CyclicLR(
            optimizer,
            base_lr=args.pop("base_lr") if "base_lr" in args else  1e-5,
            max_lr=args.pop("max_lr") if "max_lr" in args else  1e-3,
            mode='triangular',
            **args
        )
    elif t.train_lr_scheduler == "one_cycle":
        print("Using OneCycleLR")
        return OneCycleLR(
            optimizer,
            max_lr=args.pop("max_lr") if "max_lr" in args else  1e-3,
            total_steps=t.train_iterations,
            **args
        )
    
    return get_scheduler(
        name=t.train_lr_scheduler,
        optimizer=optimizer,
        step_rules=t.train_lr_step_rules,
        num_warmup_steps=t.train_lr_warmup_steps if t.train_lr_warmup_steps > 0 else 0,
        num_training_steps=t.train_iterations,
        num_cycles=t.train_lr_scheduler_num_cycles if t.train_lr_scheduler_num_cycles > 0 else 1,
        power=t.train_lr_scheduler_power if t.train_lr_scheduler_power > 0 else 1.0,
        **t.train_lr_scheduler_settings
    )

def load_torch_file(ckpt, safe_load=True, device=None): # Default to safe_load=True for CLI
    # Removed checkpoint_pickle import
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            # Removed pickle_module=checkpoint_pickle
            pl_sd = torch.load(ckpt, map_location=device)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd   


VAE_CONFIG_SD1 ={
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.6.0",
  "act_fn": "silu",
  "block_out_channels": [128,256,512,512],
  "down_block_types": ["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
  "in_channels": 3,
  "latent_channels": 4,
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 256,
  "up_block_types": ["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
}

VAE_CONFIG_SD2 = {
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.8.0",
  "_name_or_path": "hf-models/stable-diffusion-v2-768x768/vae",
  "act_fn": "silu",
  "block_out_channels": [128,256,512,512],
  "down_block_types": ["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
  "in_channels": 3,
  "latent_channels": 4,
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 768,
  "up_block_types": ["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
}

VAE_CONFIG_SDXL={
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.20.0.dev0",
  "_name_or_path": "",
  "act_fn": "silu",
  "block_out_channels": [128,256,512,512],
  "down_block_types": ["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
  "force_upcast": True,
  "in_channels": 3,
  "latent_channels": 4,
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 1024,
  "scaling_factor": 0.13025,
  "up_block_types": ["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
}

VAE_CONFIG_SD3 = {
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.29.0.dev0",
  "act_fn": "silu",
  "block_out_channels": [128,256,512,512],
  "down_block_types": ["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
  "force_upcast": True,
  "in_channels": 3,
  "latent_channels": 16,
  "latents_mean": None,
  "latents_std": None,
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 1024,
  "scaling_factor": 1.5305,
  "shift_factor": 0.0609,
  "up_block_types": ["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
  "use_post_quant_conv": False,
  "use_quant_conv": False
}

VAE_CONFIG_FLUX = {
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.30.0.dev0",
  "_name_or_path": "../checkpoints/flux-dev",
  "act_fn": "silu",
  "block_out_channels": [128,256,512,512],
  "down_block_types": ["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
  "force_upcast": True,
  "in_channels": 3,
  "latent_channels": 16,
  "latents_mean": None,
  "latents_std": None,
  "layers_per_block": 2,
  "mid_block_add_attention": True,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 1024,
  "scaling_factor": 0.3611,
  "shift_factor": 0.1159,
  "up_block_types": ["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
  "use_post_quant_conv": False,
  "use_quant_conv": False
}
VAE_CONFIGS = [VAE_CONFIG_SD1, VAE_CONFIG_SD2, VAE_CONFIG_SDXL, VAE_CONFIG_SD3, VAE_CONFIG_FLUX]

from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

#### Load VAE ####################################################    
def load_VAE(t, path):
    vae_config = VAE_CONFIGS[t.sdver]
    state_dict = load_torch_file(path,safe_load=True)
    state_dict = convert_ldm_vae_checkpoint(state_dict, vae_config)
    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(state_dict, strict=False)
    vae.eval()
    print("VAE is loaded from", path)
    return vae