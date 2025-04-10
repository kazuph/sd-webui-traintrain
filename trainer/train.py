import os
import csv
import random
import time
import numpy
import gc
import json
from PIL import Image
import traceback
import torch
from torch.nn import ModuleList
from tqdm import tqdm
import safetensors.torch

from pprint import pprint
from accelerate.utils import set_seed

# Removed WebUI specific imports and standalone/forge logic
# Assuming standalone operation for CLI
from trainer.lora import LoRANetwork, LycorisNetwork # Adjusted import path
from trainer import trainer, dataset # Adjusted import path
MODEL_LIST = ["SD1", "SD2", "SDXL", "SD3", "FLUX","REFINER", "UNKNOWN"]
MAX_DENOISING_STEPS = 1000
ML = "LoRA"
MD = "Difference"

jsonspath = trainer.jsonspath
logspath = trainer.logspath
presetspath = trainer.presetspath

stoptimer = 0

CUDA = torch.device("cuda:0")

queue_list = []
current_name = None

#dfalse, mode, model, vae, *train_settings_1, *train_settings_2, *prompts, *in_images

# get_name_index 関数は all_configs が空のため不要になったので削除
def queue(*args):
    global queue_list
    name_index = get_name_index("save_lora_name") + 4
    dup = args[name_index] == current_name
    for queue in queue_list:
        if queue[name_index] == args[name_index]:
            dup = True
    if dup:
        return ("Duplicated LoRA name! Could not add to queue.")

    queue_list.append(args)
    return "Added to Queue"

def get_del_queue_list(del_name = None):
    global queue_list
    name_index = get_name_index("save_lora_name")
    out = []
    del_index = None

    for i, q in enumerate(queue_list):
        data = [*q[1:-2]]
        name = data[name_index + 3]
        data = [name] + data
        if del_name and name == del_name:
            del_index = i
        else:
            out.append(data)
    if del_index:
        del queue_list[del_index]
    return out

def setcurrentname(args):
    name_index = get_name_index("save_lora_name") + 4
    global current_name
    current_name = args[name_index]

def train(*args):
    if not args[0]:
        setcurrentname(args)
    result = train_main(*args)
    while len(queue_list) > 0:
        settings = queue_list.pop(0)
        result +="\n" + train_main(*settings)
    flush()
    return result

# *args を config_dict に変更
def train_main(jsononly_or_paths, config_dict):
    # Trainer に config_dict を渡すように変更
    # mode, modelname, vaename は config_dict から取得する
    mode = config_dict.get('mode')
    modelname = config_dict.get('model')
    vaename = config_dict.get('vae')
    t = trainer.Trainer(jsononly_or_paths, modelname, vaename, mode, config_dict)

    if type(jsononly_or_paths) is bool and jsononly_or_paths == True:
        return "Preset saved"

    if t.isfile:
        return "File exist!"

    if modelname == "":
        return "No Model Selected."

    print(" Start Training!")

    # --- Standalone/CLI Model Loading Logic ---
    # Determine absolute model path
    if not os.path.isabs(modelname):
        checkpoint_filename = os.path.join(t.models_dir, modelname) if hasattr(t, "models_dir") else modelname
    else:
        checkpoint_filename = modelname

    if not os.path.exists(checkpoint_filename):
        return f"Model file not found: {checkpoint_filename}"

    # Detect model version from state_dict
    try:
        state_dict = trainer.load_torch_file(checkpoint_filename, safe_load=True) # Use safe_load
        model_version = detect_model_version(state_dict)
        del state_dict
        flush()
        t.sd_typer(ver=model_version) # Pass detected version
    except Exception as e:
        return f"Error loading or detecting model version: {e}"

    # Determine absolute VAE path
    vae = None
    vae_path = None
    if vaename and vaename != "None":
        if not os.path.isabs(vaename):
            vae_path = os.path.join(t.vae_dir, vaename) if hasattr(t, "vae_dir") else vaename
        else:
            vae_path = vaename
        if not os.path.exists(vae_path):
            print(f"Warning: VAE file not found at {vae_path}, proceeding without external VAE.")
            vae_path = None # Reset path if not found
    # --- End Standalone/CLI Model Loading Logic ---
    if not vae:
        vae = trainer.load_VAE(t, vae_path) if vae_path is not None else None

    if t.is_sdxl: 
        text_model, unet, vae = trainer.load_checkpoint_model_xl(checkpoint_filename, t, vae = vae)
    else:
        text_model, unet, vae = trainer.load_checkpoint_model(checkpoint_filename, t, vae = vae)
    
    unet.to(CUDA, dtype=t.train_model_precision)
    try:
        unet.enable_xformers_memory_efficient_attention()
        print("Enabling Xformers")
    except:
        print("Failed to enable Xformers")

    unet.requires_grad_(False)
    unet.eval()

    text_model.to(device = CUDA, dtype = t.train_model_precision)
    text_model.requires_grad_(False)
    text_model.eval()

    if t.use_gradient_checkpointing:
        unet.train()
        unet.enable_gradient_checkpointing()
        text_model.train()
        text_model.gradient_checkpointing_enable()

    t.unet = unet
    t.text_model = text_model

    vae = vae.to(CUDA, dtype=t.train_model_precision)
    t.vae = vae
    
    t.text2cond = text2cond
    t.image2latent = image2latent

    t.noise_scheduler = trainer.load_noise_scheduler("ddpm", t.model_v_pred)

    t.a = trainer.make_accelerator(t)

    t.unet = t.a.prepare(t.unet)
    t.text_model.text_encoders = ModuleList([t.a.prepare(te) if te is not None else None for te in t.text_model.text_encoders])

    if 0 > t.train_seed: t.train_seed = random.randint(0, 2**32)
    set_seed(t.train_seed)
    makesavelist(t)
    
    # --- Standalone/CLI Text Encoding ---
    # Encode prompts directly using the loaded text_model
    print("Encoding prompts...")
    with torch.no_grad(), t.a.autocast():
        try:
            t.orig_cond, t.orig_vector = t.text_model.encode_text([t.prompts[0]]) # Pass as list
            t.targ_cond, t.targ_vector = t.text_model.encode_text([t.prompts[1]]) # Pass as list
            t.un_cond, t.un_vector = t.text_model.encode_text([t.prompts[2]])     # Pass as list
        except Exception as e:
            return f"Error encoding prompts: {e}"
    # --- End Standalone/CLI Text Encoding ---
    del vae, text_model, unet

    try:
        if t.mode == ML:
            result = train_lora(t)
        elif t.mode == "iLECO":
            result = train_leco(t)
        elif t.mode == "Difference":
            result = train_diff(t)
        elif t.mode ==  "ADDifT" or t.mode == "Multi-ADDifT":
            result = train_diff2(t)
        else:
            result = "Test mode"

        print("Done.")

    except Exception as e:
        print(traceback.format_exc())
        result =  f"Error: {e}"
    
    t.unet.to("cpu")   
    t.unet = None
    del t
    flush()
    
    # Removed WebUI specific model reloading logic
    return result

def train_lora(t):
    global stoptimer
    stoptimer = 0

    t.a.print("Preparing image latents and text-conditional...")
    dataloaders = dataset.make_dataloaders(t)
    t.dataloader = dataset.ContinualRandomDataLoader(dataloaders)
    t.dataloader = (t.a.prepare(t.dataloader))

    t.a.print("Train LoRA Start")
    
    network, optimizer, lr_scheduler = create_network(t)

    if not t.dataloader.data:
        return "No data!"

    loss_ema = None
    loss_velocity = None

    del t.vae
    if "BASE" not in t.network_blocks:
        del t.text_model
    flush()

    pbar = tqdm(range(t.train_iterations))
    while t.train_iterations >= pbar.n:
        for batch in t.dataloader:
            for i in range(t.train_repeat):
                latents = batch["latent"].to(CUDA, dtype=t.train_lora_precision)
                conds1 = batch["cond1"]
                conds2 = batch["cond2"] if "cond2" in batch else None

                noise = torch.randn_like(latents)

                batch_size = latents.shape[0]

                timesteps = torch.randint(t.train_min_timesteps, t.train_max_timesteps, ((1 if t.train_fixed_timsteps_in_batch else batch_size),),device=CUDA) 
                timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))

                noisy_latents = t.noise_scheduler.add_noise(latents, noise, timesteps)

                with network, t.a.autocast():
                    if isinstance(conds1[0], str):
                        conds1, conds2 = t.text_model.encode_text(conds1)

                    conds1 = conds1.to(CUDA, dtype=t.train_lora_precision) 
                    if conds2 is not None:
                        conds2 = conds2.to(CUDA, dtype=t.train_lora_precision)

                    added_cond_kwargs = get_added_cond_kwargs(t, conds2, batch_size, size = [*latents.shape[2:4]])
                    noise_pred = t.unet(noisy_latents, timesteps, conds1, added_cond_kwargs = added_cond_kwargs).sample

                if t.model_v_pred:
                    noise = t.noise_scheduler.get_velocity(latents, noise, timesteps)
                    
                loss, loss_ema, loss_velocity = process_loss(t, noise_pred, noise, timesteps, loss_ema, loss_velocity)

                c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]
                pbar.set_description(f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Current LR: "+", ".join(c_lrs)+ f", Epoch: {t.dataloader.epoch}")   
                pbar.update(1)

                if t.logging_save_csv:
                    savecsv(t, pbar.n, loss_ema, [x.cpu().item() if isinstance(x, torch.Tensor)  else x for x in lr_scheduler.get_last_lr()],t.csvpath)

                t.a.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                del noise_pred, noise

                flush()

                #print(network.check_weight(True)[:10])
                #print(network.check_weight(False)[:10])
                
                result = finisher(network, t, pbar.n)
                if result is not None:
                    return result

            if pbar.n >=t.train_iterations:
                break

    return savecount(network, t, 0)

def train_leco(t):
    global stoptimer
    stoptimer = 0

    del t.vae
    if "BASE" not in t.network_blocks:
        del t.text_model
    flush()

    network, optimizer, lr_scheduler = create_network(t)

    t.orig_cond = torch.cat([t.orig_cond] * t.train_batch_size)
    t.targ_cond = torch.cat([t.targ_cond] * t.train_batch_size)

    if t.orig_vector is not None:
        t.orig_vector = torch.cat([t.orig_vector] * t.train_batch_size)
        t.targ_vector = torch.cat([t.targ_vector] * t.train_batch_size)

    height, width = t.image_size

    latents = torch.randn((t.train_batch_size, 4, height // 8, width // 8), device=CUDA,dtype = t.train_model_precision)

    loss_ema = None
    loss_velocity = None

    pbar = tqdm(range(t.train_iterations))
    while t.train_iterations >= pbar.n:
        with torch.no_grad(), t.a.autocast():                
            timesteps = torch.randint(t.train_min_timesteps, t.train_max_timesteps, (t.train_batch_size,),device=CUDA)
            timesteps = timesteps.long()
            added_cond_kwargs = get_added_cond_kwargs(t, t.targ_vector, t.train_batch_size)
            targ_pred = t.unet(latents, timesteps, t.targ_cond, added_cond_kwargs = added_cond_kwargs).sample
        
        added_cond_kwargs = get_added_cond_kwargs(t, t.orig_vector, t.train_batch_size)

        with network, t.a.autocast():
            orig_pred = t.unet(latents, timesteps, t.orig_cond, added_cond_kwargs = added_cond_kwargs).sample

        loss, loss_ema, loss_velocity = process_loss(t, orig_pred, targ_pred, timesteps, loss_ema, loss_velocity)

        c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]

        pbar.set_description(f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Current LR: "+", ".join(c_lrs))   
        pbar.update(1)
        if t.logging_save_csv:
            savecsv(t, pbar.n, loss_ema, [x.cpu().item() if isinstance(x, torch.Tensor)  else x for x in lr_scheduler.get_last_lr()],t.csvpath)

        t.a.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        flush()

        result = finisher(network, t, pbar.n)
        if result is not None:
            return result

    return savecount(network, t, 0)


def train_diff(t):
    global stoptimer
    stoptimer = 0

    t.orig_latent = image2latent(t,t.images[0]).to(t.train_model_precision).repeat_interleave(t.train_batch_size,0)
    t.targ_latent = image2latent(t,t.images[1]).to(t.train_model_precision)

    del t.vae
    if "BASE" not in t.network_blocks:
        del t.text_model
    flush()

    print("Copy Machine Start")
    t.image_size = [*t.orig_latent.shape[2:4]]

    orig_network, result = make_diff_lora(t, True) 
    if "Stopped" in result:
        return "Stopped"

    orig_network.eval()
    orig_network.requires_grad_(False)

    
    if t.diff_1st_pass_only:
        return result

    del t.a
    print("Target LoRA Start")
    t.setpass(1)

    t.a = trainer.make_accelerator(t)
    if 0 > t.train_seed: t.train_seed = random.randint(0, 2**32)
    set_seed(t.train_seed)
    makesavelist(t)

    t.targ_latent = t.targ_latent.repeat_interleave(t.train_batch_size,0)
    t.image_size = [*t.targ_latent.shape[2:4]]
    t.diff_load_1st_pass = ""
    with orig_network:
        _, result = make_diff_lora(t, False)
    return result

def make_diff_lora(t, copy):
    image_latent = t.orig_latent if copy else t.targ_latent
    batch_size = image_latent.shape[0]
    network, optimizer, lr_scheduler = create_network(t)
    added_cond_kwargs = get_added_cond_kwargs(t, torch.cat([t.targ_vector] * batch_size), batch_size) if t.targ_vector is not None else None 

    if t.diff_load_1st_pass and copy:
        return network, ""

    loss_ema = None
    loss_velocity = None

    pbar = tqdm(range(t.train_iterations))
    while t.train_iterations >= pbar.n:
        optimizer.zero_grad()
        noise = torch.randn_like(image_latent)

        timesteps = torch.randint(t.train_min_timesteps, t.train_max_timesteps, ((1 if t.train_fixed_timsteps_in_batch else batch_size),),device=CUDA) 
        timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))

        noisy_latents = t.noise_scheduler.add_noise(image_latent, noise, timesteps)

        with network, t.a.autocast():
            noise_pred = t.unet(noisy_latents, timesteps, torch.cat([t.orig_cond] * batch_size), added_cond_kwargs = added_cond_kwargs).sample

        loss, loss_ema, loss_velocity = process_loss(t, noise_pred, noise, timesteps, loss_ema, loss_velocity, copy)

        c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]

        pbar.set_description(f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Loss Velosity: {loss_velocity * 1000:.4f}, Current LR: "+", ".join(c_lrs))   
        pbar.update(1)
        if t.logging_save_csv:
            savecsv(t, pbar.n, loss_ema, [x.cpu().item() if isinstance(x, torch.Tensor)  else x for x in lr_scheduler.get_last_lr()],t.csvpath, copy = copy)

        t.a.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        flush()
        
        result = finisher(network, t, pbar.n, copy = copy)
        if result is not None:
            del optimizer, lr_scheduler
            return network, result

    result = savecount(network, t, 0, copy = copy)

    return network, result


def train_diff2(t):
    global stoptimer
    stoptimer = 0

    if t.mode == "ADDifT":
        t.orig_latent = image2latent(t,t.images[0]).to(t.train_model_precision)
        t.targ_latent = image2latent(t,t.images[1]).to(t.train_model_precision)
        data = dataset.LatentsConds(t, [([t.orig_latent, None, t.orig_cond, t.targ_vector],[t.targ_latent,None, t.targ_cond, t.targ_vector])])
        dataloaders = [dataset.DataLoader(data, batch_size=t.train_batch_size, shuffle=True)]
    else:
        t.a.print("Preparing image latents and text-conditional...")
        dataloaders = dataset.make_dataloaders(t)
    t.dataloader = dataset.ContinualRandomDataLoader(dataloaders)
    t.dataloader = (t.a.prepare(t.dataloader))

    t.a.print("Train ADDifT Start")

    if not t.dataloader.data:
        return "No data!"

    del t.vae
    if "BASE" not in t.network_blocks:
        del t.text_model
    flush()
  
    network, optimizer, lr_scheduler = create_network(t)

    loss_ema = None
    noise = None
    loss_velocity = None

    pbar = tqdm(range(t.train_iterations))
    epoch = 0
    while t.train_iterations >= pbar.n + 1:
        for batch in t.dataloader:
            orig_latent = batch["orig_latent"]
            targ_latent = batch["targ_latent"]

            batch_size = orig_latent.shape[0]
 
            orig_conds1 = batch["orig_conds1"] if "orig_conds1" in batch else torch.cat([t.orig_cond] * batch_size)
            orig_conds2 = batch["orig_conds2"] if "orig_conds2" in batch else (torch.cat([t.orig_vector] * batch_size) if isinstance(t.orig_vector, torch.Tensor) else None)
            targ_conds1 = batch["targ_conds1"] if "targ_conds1" in batch else torch.cat([t.targ_cond] * batch_size)
            targ_conds2 = batch["targ_conds2"] if "targ_conds2" in batch else (torch.cat([t.targ_vector] * batch_size) if isinstance(t.targ_vector, torch.Tensor) else None)
            
            orig_added_cond_kwargs = get_added_cond_kwargs(t, orig_conds2, batch_size, size = [*orig_latent.shape[2:4]]) if orig_conds2 is not None else None 
            targ_added_cond_kwargs = get_added_cond_kwargs(t, targ_conds2, batch_size, size = [*targ_latent.shape[2:4]]) if targ_conds2 is not None else None 

            optimizer.zero_grad()
            noise = torch.randn_like(orig_latent) 

            turn = pbar.n % 2 == 0 

            #t.train_min_timesteps = int(500 * (pbar.n * 2 / t.train_iterations)) if 1 > pbar.n * 2 / t.train_iterations else 500
            #t.train_max_timesteps = int(750 + 250 * (1 - (pbar.n * 2/ (t.train_iterations * 2)))) if 1 > pbar.n * 2 / t.train_iterations else 750
            #print(t.train_min_timesteps, t.train_max_timesteps)
            if turn:
                span = (t.train_max_timesteps - t.train_min_timesteps) /10
                index = pbar.n % 10 + 1
                time_min = t.train_min_timesteps + span * index
                time_max = t.train_min_timesteps + span * (index + 1)

                if not (time_min > time_max):
                    time_max = time_min + 1

            timesteps = torch.randint(int(999 if time_min > 999 else time_min), int(time_max if 1000 > time_max else 1000), ((1 if t.train_fixed_timsteps_in_batch else batch_size),),device=CUDA) 
            timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))

            if 0 > t.diff_alt_ratio and not turn:
                targ_latent = orig_latent = noise

            orig_noisy_latents = t.noise_scheduler.add_noise(orig_latent if turn else targ_latent, noise, timesteps)
            targ_noisy_latents = t.noise_scheduler.add_noise(targ_latent if turn else orig_latent, noise, timesteps)

            with torch.no_grad(), t.a.autocast(): 
                orig_noise_pred = t.unet(orig_noisy_latents, timesteps, orig_conds1, added_cond_kwargs = orig_added_cond_kwargs).sample
            
            # 乗数を小さくして数値安定性を試す (0.25 -> 0.1)
            network.set_multiplier(0.25 if turn else - 0.25 * abs(t.diff_alt_ratio)) # 0.25に戻す

            with t.a.autocast():
                targ_noise_pred = t.unet(targ_noisy_latents, timesteps, targ_conds1, added_cond_kwargs = targ_added_cond_kwargs).sample
            network.set_multiplier(0)

            if t.diff_use_diff_mask and "mask" in batch:
                targ_noise_pred = targ_noise_pred * batch["mask"].to(CUDA) 
                orig_noise_pred = orig_noise_pred * batch["mask"].to(CUDA)   

            loss, loss_ema, loss_velocity = process_loss(t, targ_noise_pred, orig_noise_pred, timesteps, loss_ema, loss_velocity)

            c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]

            pbar.set_description(f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Loss Velosity: {loss_velocity * 1000:.4f}, Current LR: "+", ".join(c_lrs) + f", Epoch: {epoch}")   
            pbar.update(1)
            if t.logging_save_csv:
                savecsv(t, pbar.n, loss_ema, [x.cpu().item() if isinstance(x, torch.Tensor)  else x for x in lr_scheduler.get_last_lr()],t.csvpath)
            
            optimizer.zero_grad()
            t.a.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            flush()
            
            result = finisher(network, t, pbar.n)
            if result is not None:
                return result

        epoch += 1
        
    result = savecount(network, t, 0)
    return result

#### Prepare LoRA, Optimizer, lr_scheduler, Save###############################################
def flush():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

def create_network(t):
    network = load_network(t)
    optimizer= trainer.get_optimizer(t.train_optimizer, network.prepare_optimizer_params(), t.train_learning_rate, t.train_optimizer_settings, network)

    t.is_schedulefree = t.train_optimizer.endswith("schedulefree".lower())

    if t.is_schedulefree:
        optimizer.train()
    else:
        lr_scheduler = trainer.load_lr_scheduler(t, optimizer)

    print(f"Optimizer : {type(optimizer).__name__}")
    print(f"Optimizer Settings : {t.train_optimizer_settings}")

    network, optimizer, lr_scheduler = t.a.prepare(network, optimizer, None if t.is_schedulefree else lr_scheduler)

    return network, optimizer, DummyScheduler(optimizer) if t.is_schedulefree else lr_scheduler

class DummyScheduler():
    def __init__(self, optimizer):
        self.optimizer = optimizer
        pass

    def get_last_lr(self):
        return [p["scheduled_lr"] for p in self.optimizer.param_groups]

    def step(self):
        pass

def load_network(t):
    # network_type が存在するか確認し、存在しない場合はデフォルト値（例: 'lora'）を設定するか、エラーを出す
    # trainer モジュールから lora をインポートしておく必要がある
    from trainer import lora
    network_type = getattr(t, 'network_type', 'lora').lower() # デフォルトを 'lora' と仮定

    # lierla と c3lier は LoRANetwork を使用
    if network_type in ['lierla', 'c3lier', 'lora']: # lora も LoRANetwork を使うように明示
        print(f"Loading LoRANetwork for network_type: {network_type}")
        # LoRANetwork の初期化時に t を渡す
        return LoRANetwork(t).to(CUDA, dtype=t.train_lora_precision)
    # LyCORIS系のネットワークタイプ (loha, locon, ia3 など) は LycorisNetwork を使用
    elif network_type in lora.network_module_dict: # lora.py の辞書を参照
        print(f"Loading LycorisNetwork for network_type: {network_type}")
        # LycorisNetwork の初期化時に t を渡す
        return LycorisNetwork(t).to(CUDA, dtype=t.train_lora_precision)
    else:
        # 未知またはデフォルトのタイプは LoRANetwork を使うか、エラーとする
        print(f"Warning: Unknown or unsupported network_type '{network_type}'. Falling back to LoRANetwork.")
        # デフォルトとして LoRANetwork を返すか、エラー処理を行う
        return LoRANetwork(t).to(CUDA, dtype=t.train_lora_precision)
        # または raise ValueError(f"Unsupported network_type: {network_type}")

def stop_time(save):
    global stoptimer
    stoptimer = 2 if save else 1

def finisher(network, t, i, copy = False):
    if t.save_list and i >= t.save_list[0]:
        savecount(network, t, t.save_list.pop(0), copy)

    if stoptimer > 0:
        if stoptimer > 1:
            result = ". " + savecount(network, t, i, copy)
        else:
            result = ""
        return "Stopped" + result

def savecount(network, t, i, copy = False):
    if t.metadata == {}:
       metadator(t)
    if copy and not t.diff_save_1st_pass:
        return "Not save copy"
    add = "_copy" if copy else ""
    add = f"{add}_{i}steps" if i > 0 else add
    filename = os.path.join(t.save_dir, f"{t.save_lora_name}{add}.safetensors")
    print(f" Saving to {filename}")
    metaname = f"{t.save_lora_name}{add}"
    filename = network.save_weights(filename, t, metaname)
    return f"Successfully created to {filename}"

def makesavelist(t):
    if t.save_per_steps > 0:
        t.save_list = [x * t.save_per_steps for x in range(1, t.train_iterations // t.save_per_steps + 1)]
        if t.train_iterations in t.save_list:
            t.save_list.remove(t.train_iterations)
    else:
        t.save_list = []

def process_loss(t, original, target, timesteps, loss_ema, loss_velocity, copy = False):
    if t.train_loss_function == "MSE":
        loss = torch.nn.functional.mse_loss(original.float(), target.float(), reduction="none")
    if t.train_loss_function == "L1":
        loss = torch.nn.functional.l1_loss(original.float(), target.float(), reduction="none")
    if t.train_loss_function == "Smooth-L1":
        loss = torch.nn.functional.smooth_l1_loss(original.float(), target.float(), reduction="none")
    loss = loss.mean([1, 2, 3])

    if t.train_snr_gamma > 0:
        loss = apply_snr_weight(loss, timesteps, t.noise_scheduler, t.train_snr_gamma)

    loss = loss.mean()

    if loss_ema is None:
        loss_ema = loss.item()
        loss_velocity = 0
    else:
        loss_velocity = loss_velocity * 0.9 + (loss_ema - (loss_ema * 0.9 + loss.item() * 0.1)) * 0.1
        loss_ema = loss_ema * 0.9 + loss.item() * 0.1
    
    return loss, loss_ema, loss_velocity

#### Anti-Overfitting functions ####################################################
def apply_snr_weight(loss, timesteps, noise_scheduler, gamma):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    gamma_over_snr = torch.div(torch.ones_like(snr) * gamma, snr)
    snr_weight = torch.minimum(gamma_over_snr, torch.ones_like(gamma_over_snr)).float()  # from paper
    loss = loss * snr_weight
    return loss

#### Encode Latent, Embeddings ####################################################
def image2latent(t,image):
    if isinstance(image, str):
        with Image.open(image) as img:
            image = img.convert("RGB")
    image = numpy.array(image)
    image = image.astype(numpy.float32) / 255.0
    image = numpy.moveaxis(image, 2, 0)
    image = torch.from_numpy(image).unsqueeze(0)
    image = image * 2 - 1
    # train_VAE_precision の代わりに train_model_precision を使用
    image = image.to(CUDA,dtype=t.train_VAE_precision) # train_VAE_precision を使うように戻す
    with torch.no_grad():
        # train_VAE_precision の代わりに train_model_precision を使用
        t.vae.to(dtype=t.train_VAE_precision) # train_VAE_precision を使うように戻す
        latent = t.vae.encode(image)
        if isinstance(latent, torch.Tensor):
            return ((latent - t.vae_shift_factor) * t.vae_scale_factor)
        else:
            return ((latent.latent_dist.sample() - t.vae_shift_factor) * t.vae_scale_factor)

def text2cond(t, prompt):
    # This function is now redundant as encoding happens directly in train_main
    # Kept for potential future use or refactoring, but should not be called in current CLI flow.
    print("Warning: text2cond function called, but should be handled directly by TextModel.encode_text")
    with torch.no_grad(), t.a.autocast():
        cond, vector = t.text_model.encode_text([prompt]) # Pass as list
    return cond, vector

# Removed SdConditioning class (WebUI specific)
def get_added_cond_kwargs(t, projection, batch_size, size = None):
    size = size if size is not None else t.image_size
    size_condition = list(size + [0, 0] + size)
    size_condition = torch.tensor([size_condition], dtype=t.train_model_precision, device=CUDA).repeat(batch_size, 1)
    if projection is not None:
        return {"text_embeds": projection, "time_ids": size_condition}
    else:
        return None

#### Debug, Logging ####################################################
def check_requires_grad(model: torch.nn.Module):
    for name, module in list(model.named_modules())[:5] + list(model.named_modules())[5:]:
        if len(list(module.parameters())) > 0:
            print(f"Module: {name}")
            for name, param in list(module.named_parameters())[:2]:
                print(f"    Parameter: {name}, Requires Grad: {param.requires_grad}")

def check_training_mode(model: torch.nn.Module):
    for name, module in list(model.named_modules())[:5] + list(model.named_modules())[5:]:
        print(f"Module: {name}, Training Mode: {module.training}")

CSVHEADS = ["network_rank", "network_alpha", "train_learning_rate", "train_iterations", "train_lr_scheduler", "model_version", "train_optimizer", "save_lora_name"]

def savecsv(t, step, loss, lr, csvpath, copy=False):
    header = []
    for key in CSVHEADS:
        header.append([key, getattr(t, key, "")])

    header.append(["Step", "Loss"] + ["Learning Rate " + str(i+1) for i in range(len(lr))])

    if copy:
        csvpath = csvpath.replace(".csv", "_copy.csv")
    
    directory = os.path.dirname(csvpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_exists = os.path.isfile(csvpath)

    with open(csvpath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            for head in header:
                writer.writerow(head)
        writer.writerow([step, loss] + lr)

#### Metadata ####################################################
def metadator(t):
    t.metadata = {
        "ss_session_id": random.randint(0, 2**32),  # random integer indicating which group of epochs the model came from
        "ss_training_started_at": time.time(),  # unix timestamp
        "ss_output_name": t.save_lora_name,
        "ss_learning_rate": t.train_learning_rate,
        "ss_max_train_steps": t.train_iterations,
        "ss_lr_warmup_steps": t.train_lr_warmup_steps,
        "ss_lr_scheduler": t.train_lr_scheduler,
        "ss_network_module": "network.lora",
        "ss_network_dim": t.network_rank,  # None means default because another network than LoRA may have another default dim
        "ss_network_alpha": t.network_alpha,  # some networks may not have alpha
        "ss_mixed_precision": t.train_lora_precision,
        "ss_lr_step_rules":t.train_lr_step_rules,
        "ss_lr_warmup_steps":t.train_lr_warmup_steps,
        "ss_lr_scheduler_num_cycles": t.train_lr_scheduler_num_cycles,
        "ss_lr_scheduler_power": t.train_lr_scheduler_power,
        "ss_v2": bool(t.is_sd2),
        "ss_base_model_version": t.model_version,
        "ss_seed": t.train_seed,
        "ss_optimizer": t.train_optimizer,
        "ss_min_snr_gamma": t.train_snr_gamma,
        "ss_tag_frequency": json.dumps({1:t.count_dict})
    }
    
#### StandAlone ####################################################
def detect_model_version(state_dict):
    flux_test_key = "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"
    sd3_test_key = "model.diffusion_model.final_layer.adaLN_modulation.1.bias"
    legacy_test_key = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"

    model_type = "-"
    if legacy_test_key in state_dict:
        match state_dict[legacy_test_key].shape[1]:
            case 768:
                return 0
            case 1024:
                return 1
            case 1280:
                return 5     # sdxl refiner model
            case 2048:
                return 2
    elif flux_test_key in state_dict:
        return 4
    elif sd3_test_key in state_dict:
        return 3
    else:
        return -1