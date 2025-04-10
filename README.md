# CLI Usage

## Using uv

```bash
uv venv
uv sync
uv run python cli.py --mode ADDifT --config presets/ADDifT_Action_XL.json --model <path_to_your_model> --orig_image inputs/<your_original_image> --targ_image inputs/<your_target_image> --output_name outputs/<your_output_name> --iterations 50 --save-overwrite
```

## Using Docker Compose

First, build the image:
```bash
docker compose build
```

Then, run the training:
```bash
docker compose run --rm app python cli.py --mode ADDifT --config presets/ADDifT_Action_XL.json --model <path_to_your_model_in_container> --orig_image inputs/<your_original_image> --targ_image inputs/<your_target_image> --output_name outputs/<your_output_name> --iterations 50 --save-overwrite
```

**Note:** Replace placeholders like `<path_to_your_model>`, `<your_original_image>`, `<your_target_image>`, `<your_output_name>`, and `<path_to_your_model_in_container>` with your actual file paths and names.

---

## CLI Options

Here is a list of all available command-line options. CLI arguments will override any corresponding settings provided in a JSON configuration file (`--config`).

| Option                       | Description                                       | Type      | Default        | Notes                                                                 |
| :--------------------------- | :------------------------------------------------ | :-------- | :------------- | :-------------------------------------------------------------------- |
| `--mode`                     | Training mode.                                    | string    | (Required)     | Choices: `LoRA`, `iLECO`, `Difference`, `ADDifT`, `Multi-ADDifT`        |
| `--model`                    | Path to the base model file (.safetensors or .ckpt). | string    | (Required)     |                                                                       |
| `--output-name`              | Filename for the output LoRA (without extension). | string    | (Required)     | Corresponds to `save_lora_name` internally.                           |
| `--data-dir`                 | Path to the directory containing training images. | string    | (Required for LoRA, Multi-ADDifT) | Corresponds to `lora_data_directory` internally.                      |
| `--orig-prompt`              | Original prompt.                                  | string    | (Required for iLECO) |                                                                       |
| `--targ-prompt`              | Target prompt.                                    | string    | (Required for iLECO) |                                                                       |
| `--orig-image`               | Path to the original image.                       | string    | (Required for Difference, ADDifT) |                                                                       |
| `--targ-image`               | Path to the target image.                         | string    | (Required for Difference, ADDifT) |                                                                       |
| `--config`                   | Path to a JSON configuration file.                | string    | `None`         | Overrides default settings. CLI arguments override JSON values.       |
| `--vae`                      | Path to the VAE file (optional).                  | string    | `None`         |                                                                       |
| `--network-type`             | Network type.                                     | string    | `lierla`       | Choices: `lierla`, `c3lier`, `loha`. Overrides JSON.                  |
| `--rank`                     | Network rank (dimension).                         | int       | `16`           | Corresponds to `network_rank`. Overrides JSON.                        |
| `--alpha`                    | Network alpha (reduction width).                  | float     | `8.0`          | Corresponds to `network_alpha`. Overrides JSON.                       |
| `--diff-target-name`         | Suffix for target images in Multi-ADDifT mode.    | string    | `""`           | Overrides JSON.                                                       |
| `--lora-trigger-word`        | Trigger word prepended to captions.               | string    | `""`           | Overrides JSON.                                                       |
| `--image-size`               | Training resolution (height, width).              | string    | `"512,512"`    | Comma-separated. Overrides JSON.                                      |
| `--iterations`               | Number of training iterations.                    | int       | `1000`         | Corresponds to `train_iterations`. Overrides JSON.                    |
| `--train-batch-size`         | Batch size for training.                          | int       | `2`            | Overrides JSON.                                                       |
| `--lr`                       | Learning rate.                                    | float     | `1e-4`         | Corresponds to `train_learning_rate`. Overrides JSON.                 |
| `--train-optimizer`          | Optimizer to use.                                 | string    | `AdamW`        | See `trainer.py` for choices. Overrides JSON.                         |
| `--train-optimizer-settings` | Additional optimizer settings (key=value pairs).  | string    | `""`           | Semicolon or newline separated. e.g., "weight_decay=0.01". Overrides JSON. |
| `--train-lr-scheduler`       | Learning rate scheduler.                          | string    | `cosine`       | See `trainer.py` for choices. Overrides JSON.                         |
| `--train-lr-scheduler-settings` | Additional scheduler settings (key=value pairs). | string    | `""`           | Semicolon or newline separated. Overrides JSON.                       |
| `--[no-]use-gradient-checkpointing` | Enable/disable gradient checkpointing.        | boolean   | `False`        | Overrides JSON.                                                       |
| `--network-blocks`           | Specify layers to train (space-separated).        | list      | (All Blocks)   | Choices: `BASE`, `IN00`...`M00`...`OUT11`. Overrides JSON.           |
| `--network-conv-rank`        | Convolutional layer rank (for c3lier, loha).      | int       | `0`            | `0` uses `network_rank`. Overrides JSON.                              |
| `--network-conv-alpha`       | Convolutional layer alpha (for c3lier, loha).     | float     | `0.0`          | `0` uses `network_alpha`. Overrides JSON.                             |
| `--network-resume`           | Path to LoRA file to resume training from.        | string    | `""`           | Overrides JSON.                                                       |
| `--[no-]network-train-text-encoder` | Train the text encoder(s).                  | boolean   | `False`        | Overrides JSON.                                                       |
| `--network-element`          | Detailed training target.                         | string    | `Full`         | Choices: `Full`, `CrossAttention`, `SelfAttention`. Overrides JSON.   |
| `--network-strength`         | Network strength (usually 1.0).                   | float     | `1.0`          | Overrides JSON.                                                       |
| `--train-loss-function`      | Loss function.                                    | string    | `MSE`          | Choices: `MSE`, `L1`, `Smooth-L1`. Overrides JSON.                    |
| `--train-seed`               | Random seed for training (-1 for random).         | int       | `-1`           | Overrides JSON.                                                       |
| `--train-min-timesteps`      | Minimum timestep for training.                    | int       | `0`            | Overrides JSON.                                                       |
| `--train-max-timesteps`      | Maximum timestep for training.                    | int       | `1000`         | Overrides JSON.                                                       |
| `--train-textencoder-learning-rate` | Learning rate for text encoder(s).          | float     | `None`         | If `None`, uses main `lr`. Overrides JSON.                            |
| `--train-model-precision`    | Precision for non-training parts (UNet, VAE).     | string    | `fp16`         | Choices: `fp32`, `bf16`, `fp16`. Overrides JSON.                      |
| `--train-lora-precision`     | Precision for LoRA weights during training.       | string    | `fp32`         | Choices: `fp32`, `bf16`, `fp16`. Overrides JSON.                      |
| `--train-vae-precision`      | Precision for VAE during training.                | string    | `fp32`         | Choices: `fp32`, `bf16`, `fp16`. Overrides JSON.                      |
| `--image-buckets-step`       | Step size for image bucketing resolution.         | int       | `256`          | Overrides JSON.                                                       |
| `--image-num-multiply`       | Multiply dataset images (for small datasets).     | int       | `1`            | Overrides JSON.                                                       |
| `--image-min-length`         | Minimum image side length for bucketing.          | int       | `512`          | Overrides JSON.                                                       |
| `--image-max-ratio`          | Maximum aspect ratio allowed for bucketing.       | float     | `2.0`          | Overrides JSON.                                                       |
| `--sub-image-num`            | Number of lower-resolution copies per image.      | int       | `0`            | Overrides JSON.                                                       |
| `--[no-]image-mirroring`     | Horizontally flip images for augmentation.        | boolean   | `False`        | Overrides JSON.                                                       |
| `--[no-]image-use-filename-as-tag` | Use filename (without ext) as tag if no caption. | boolean | `False`        | Overrides JSON.                                                       |
| `--[no-]image-disable-upscale` | Disable upscaling images smaller than bucket size. | boolean | `False`        | Overrides JSON.                                                       |
| `--[no-]image-use-transparent-background-ajust` | Adjust for transparent backgrounds. | boolean | `False`        | Overrides JSON.                                                       |
| `--save-per-steps`           | Save LoRA every N steps (0 to disable).           | int       | `0`            | Overrides JSON.                                                       |
| `--save-precision`           | Precision for saving the final LoRA file.         | string    | `fp16`         | Choices: `fp32`, `bf16`, `fp16`. Overrides JSON.                      |
| `--[no-]save-overwrite`      | Overwrite existing LoRA file with the same name.  | boolean   | `False`        | Overrides JSON.                                                       |
| `--[no-]save-as-json`        | Save training settings as a JSON file.            | boolean   | `False`        | Overrides JSON.                                                       |
| `--[no-]diff-save-1st-pass`  | Save the copier LoRA in Difference mode.          | boolean   | `False`        | Overrides JSON.                                                       |
| `--[no-]diff-1st-pass-only`  | Only train the copier LoRA in Difference mode.    | boolean   | `False`        | Overrides JSON.                                                       |
| `--diff-load-1st-pass`       | Path to copier LoRA to load for Difference/ADDifT. | string    | `""`           | Overrides JSON.                                                       |
| `--[no-]diff-revert-original-target` | Swap original and target images internally. | boolean | `False`        | Overrides JSON.                                                       |
| `--[no-]diff-use-diff-mask`  | Use a difference mask during training.            | boolean   | `False`        | Overrides JSON.                                                       |
| `--[no-]diff-use-fixed-noise`| Use fixed noise for Difference/ADDifT modes.      | boolean   | `False`        | Overrides JSON.                                                       |
| `--diff-alt-ratio`           | Alternate ratio for Difference mode.              | float     | `1.0`          | Overrides JSON.                                                       |
| `--train-lr-step-rules`      | Step rules for 'step' scheduler (e.g., "10,20").  | string    | `""`           | Overrides JSON.                                                       |
| `--train-lr-warmup-steps`    | Number of warmup steps for LR scheduler.          | int       | `0`            | Overrides JSON.                                                       |
| `--train-lr-scheduler-num-cycles` | Number of cycles for cosine restart schedulers. | int       | `1`            | Overrides JSON.                                                       |
| `--train-lr-scheduler-power` | Power for polynomial LR scheduler.                | float     | `1.0`          | Overrides JSON.                                                       |
| `--train-snr-gamma`          | SNR gamma value for timestep weighting (0-20).    | float     | `5.0`          | Overrides JSON.                                                       |
| `--[no-]train-fixed-timsteps-in-batch` | Use fixed timesteps within a batch.     | boolean   | `False`        | Overrides JSON.                                                       |
| `--[no-]logging-verbose`     | Output verbose logs to the console.               | boolean   | `False`        | Overrides JSON.                                                       |
| `--[no-]logging-save-csv`    | Save training progress (step, loss, lr) to CSV.   | boolean   | `False`        | Overrides JSON.                                                       |
| `--[no-]model-v-pred`        | Use v-prediction model (SD 2.x).                  | boolean   | `False`        | Overrides JSON.                                                       |
| `--[no-]use-2nd-pass-settings` | Use separate settings for the 2nd pass (Difference). | boolean | `False`        | Overrides JSON.                                                       |

---

# TrainTrain
- This is an extension for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- You can create LoRA, iLECO, and differential LoRA.

[<img src="https://img.shields.io/badge/lang-Egnlish-red.svg?style=plastic" height="25" />](#overview)
[<img src="https://img.shields.io/badge/言語-日本語-green.svg?style=plastic" height="25" />](README_ja.md)
[<img src="https://img.shields.io/badge/Support-%E2%99%A5-magenta.svg?logo=github&style=plastic" height="25" />](https://github.com/sponsors/hako-mikan)

# Recent Update
Standalone training is now supported. For details, please refer to the [Standalone Environment Setup Repository](https://github.com/hako-mikan/traintrain-standalone).

- Added new training method ADDifT
- Added Optimizers
DAdaptAdaGrad, DAdaptAdan, DAdaptSGD, SGDNesterov8bit, Lion8bit, PagedAdamW8bit, PagedLion8bit, RAdamScheduleFree, AdamWScheduleFree, SGDScheduleFree, CAME, Tiger, AdamMini, PagedAdamW, PagedAdamW32bit, SGDNesterov
- added addtional settigs for Optimizer and lr Scheduler

# Overview
This is a tool for training LoRA for Stable Diffusion. It operates as an extension of the Stable Diffusion Web-UI and does not require setting up a training environment. It accelerates the training of regular LoRA, iLECO (instant-LECO), which speeds up the learning of LECO (removing or emphasizing a model's concept), and differential learning that creates slider LoRA from two differential images.

# Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
    - [LoRA](#lora)
    - [iLECO](#ileco)
    - [Difference](#difference)
    - [ADDifT](#addift)
    - [Multi-ADDifT](#multi-addift)
- [Settings](#settings)
    - [Mandatory Parameters](#mandatory-parameters)
    - [Optional Parameters](#optional-parameters)
- [Queue (Reserving Training)](#queue)
- [Plot](#plot)
- [Acknowledgments & References](#acknowledgments)

## Requirements
   Operates with Web-UI 1.10, latest version of Forge/reForge.

## Installation
   Enter `https://github.com/hako-mikan/sd-webui-traintrain` in the Web-UI's Install From URL and press the Install button, then restart. The first startup may take a little time (a few seconds to tens of seconds).

## Usage
   Enter the mandatory parameters for each mode and press the Start Training button to begin training. The created LoRA will be saved in the LoRA folder.
## LoRA
Learn LoRA from images.
### Input Images
   Supports `jpg`, `jpeg`, `png`, `gif`, `tif`, `tiff`, `bmp`, `webp`, `pcx`, `ico` formats. The size does not need to be the one specified by `image size`, but it will be cropped during training, so it's better to format the images to some extent to avoid inconsistencies with the captions. Images are classified by aspect ratio. For example, if you set the `image size` to 768x512, several resolution sets (buckets) will be created with a maximum pixel size of 768x512. By default, it classifies into three types: 768x512, 512x512, and 512x768, and images are sorted into the closest classification by aspect ratio. This is because the training only accepts images of the same size. During this process, images are resized and cropped. The cropping is centered on the image's center. To refine the classification, decrease the value of `image buckets step`.

### Image Resizing & Mirroring
   Training the same image repeatedly can lead to overfitting, where the image itself appears. If there are few training images, we deal with overfitting by resizing and flipping images to increase the number of training images. If you set `image size` to `768,512` and `image buckets step` to `128`, the frames `(384, 768), (512, 768), (512, 640), (512, 512), (640, 512), (768, 512), (768, 384)` are created. Additionally, setting `image min length` to `256` creates frames for resizing such as `(256, 512), (384, 640), (256, 384), (384, 512), (384, 384), (256, 256), (512, 384), (384, 256), (640, 384), (512, 256)`. Images are first sorted into normal frames, but if `sub image num` is set, they are also resized and stored in resizing frames with a similar aspect ratio. For instance, if an image is stored in a `(512, 640)` frame and `sub image num` is set to 3, it is also resized and stored in `(384, 640)`, `(256, 384)`, and `(384, 512)`. If `image mirroring` is enabled, mirrored images are also stored,

 resulting in 8 training images from one image.

### Captions, Trigger Words
   If there are `txt` or `caption` files with the same filename as the image, the text in these files is used for training. If both exist, the `txt` file takes precedence. If `trigger word` is set, it is inserted before all captions, including when there is no caption file.

### Approach to Captions
   Let's say you're training a character named A. A has twin tails, wears a blue shirt, and a red skirt. If there's a picture of A against a white background, the caption should include A's name, the direction they're facing, and that the background is white. Elements unique to A, like twin tails, blue shirt, and red skirt, shouldn't be included in the caption as they are specific to A and you want to train for them. However, direction, background, and composition, which you don't want to learn, should be included.

## iLECO
   iLECO (instant-LECO) is a faster version of LECO training, transforming the concept specified in Original Prompt closer to the concept in Target Prompt. If nothing is entered in Target Prompt, it becomes training to remove that concept.
   For example, let's erase the Mona Lisa, which appears robustly in any model. Enter "Mona Lisa" in Original Prompt and leave Target Prompt blank. It converges with about 500 `train iterations`. The value of `alpha` is usually set smaller than rank, but in the case of iLECO, a larger value than rank may be better.
![](https://github.com/hako-mikan/sd-webui-traintrain/blob/images/sample1.jpg)   
　We succeeded in erasing the Mona Lisa. Next, enter "Vincent van Gogh Sunflowers" in Target Prompt. Now, the Mona Lisa turns into sunflowers in the LoRA.
 ![](https://github.com/hako-mikan/sd-webui-traintrain/blob/images/sample2.jpg)   
   Try entering "red" in Original Prompt and "blue" in Target Prompt. You get a LoRA that turns red into blue.
 ![](https://github.com/hako-mikan/sd-webui-traintrain/blob/images/sample3.jpg) 
## Difference
   Creates LoRA from two differential images. This is known as the copy machine learning method. First, create a copy machine LoRA (which only produces the same image), then apply LoRA and train for the difference to create a differential LoRA. Set images in Original and Target. The image size should be the same.
   First, training for the copy machine begins, followed by training for the difference. For example, let's make a LoRA for closing eyes using the following two images.  
   <img src="https://github.com/hako-mikan/sd-webui-traintrain/blob/images/sample4.jpg" width="200">
   <img src="https://github.com/hako-mikan/sd-webui-traintrain/blob/images/sample5.jpg" width="200">  
   Use Difference_Use2ndPass. Set `train batch size` to 1-3. A larger value does not make much difference.   
    ![](https://github.com/hako-mikan/sd-webui-traintrain/blob/images/sample6.jpg)   
    We succeeded. Other than closing the eyes, there is almost no impact on the painting style or composition. This is because the rank(dim) is set to 4, which is small in the 2ndPass. If you set this to the same 16 as the copy machine, it will affect the painting style and composition.

## ADDifT  
Creates a LoRA from two difference images. Unlike copier learning, this method directly trains the LoRA on the differences, making it significantly faster. It does not train copier LoRAs. Set the images for `"Original"` and `"Target"` and ensure they have the same size. Properly adjusting the min/max timesteps is crucial for effective learning, depending on the target subject. For actions or decorations like opening/closing eyes, set Min = 500 and Max = 1000. For art styles, Min = 200 and Max = 400 work well. The number of training iterations should be around 30 to 100; exceeding this may lead to overfitting. The batch size should be set to 1. Although increasing the batch size is possible, reducing the number of training iterations would be necessary, so keeping a small batch size and increasing iterations generally yields better results.

## Multi-ADDift  
Creates a difference LoRA from multiple sets of two images. It follows the same directory-based approach as LoRA training, with pairs determined by file names. The training pairs are formed using images and those specified with the `"diff target name."` For example, if the `"diff target name"` is `"_closed_eyes,"` the method will pair images like `"image1.png, image2.png"` with `"image1_closed_eyes.png, image2_closed_eyes.png"` for training. As with standard LoRA training, loaded images are bucketed based on their size. For more details, refer to [Image Resizing & Mirroring](#Image-Resizing-&-Mirroring).

> [!TIP]
> If you don't have enough VRAM, enable `gradient checkpointing`. It will slightly extend the computation time but reduce VRAM usage. In some cases, activating `gradient checkpointing` and increasing the batch size can shorten the computation time. In copy machine learning, increasing the batch size beyond 3 makes little difference, so it's better to keep it at 3 or less. The batch size is the number of images learned at once, but doubling the batch size doesn't mean you can halve the `iterations`. In one learning step, the weights are updated once, but doubling the batch size does not double the number of updates, nor does it double the efficiency of learning.

## Settings
## Mandatory Parameters

| Parameter | Details |
|-----------|---------|
| network type | lierla is a standard LoRA. c3lier (commonly known as LoCON) and loha (commonly known as LyCORIS) increase the learning area. If you choose c3lier or loha, you can adjust the dimensions of the additional area by setting the `conv rank` and `conv alpha` options. |
| network rank | The size of LoRA, also known as dim. It's not good to be too large, so start around 16. |
| network alpha | The reduction width of LoRA. Usually set to the same or a smaller value than rank. For iLECO, a larger value than rank may be better. |
| lora data directory | Specifies the folder where image files for LoRA learning are stored. Subfolders are also included. |
| lora trigger word | When not using caption files, learning is performed associated with the text written here. Details in the learning section. |
| network blocks | Used for layer-specific learning. BASE refers to the TextEncoder. BASE is not used in iLECO, Difference. |
| train iterations | Number of learning iterations. 500 to 1000 is appropriate for iLECO, Difference. |
| image size | The resolution during learning. The order of height and width is only valid for iLECO. |
| train batch size | How many images are learned at once. Set to an efficient level so that VRAM does not overflow the shared memory. |
| train learning rate | The learning rate. 1e-3 to 1e-4 for iLECO, and about 1e-3 for Difference is appropriate. |
| train optimizer | Setting for the optimization function. adamw is recommended. adamw8bit reduces accuracy. Especially in Difference, adamw8bit does not work well. |
| train lr scheduler | Setting to change the learning rate during learning. Just set it to cosine. If you choose adafactor as the optimizer, the learning rate is automatically determined, and this item is deactivated. |
| save lora name | The file name when saving. If not set, it becomes untitled. |
| use gradient checkpointing | Reduces VRAM usage at the expense of slightly slower learning. |

## Optional Parameters
Optional, so they work even if not specified.
| Parameter | Details |
|-----------|---------|
| network conv rank | Rank of the conv layer when using c3lier, loha. If set to 0, the value of network rank is used. |
| network conv alpha | Reduction width of the conv layer when using c3lier, loha. If set to 0, the value of network alpha is used. |
| network element | Specifies the learning target in detail. Does not work with loha.<br>Full: Same as the normal LoRA.<br>CrossAttention: Only activates layers that process generation based on prompts.<br>SelfAttention: Only activates layers that process generation without prompts. |
| train lr step rules | Specifies the steps when the lr scheduler is set to step. |
| train lr scheduler num cycles | Number of repetitions for cosine with restart. |
| train lr scheduler power | Exponent when the lr scheduler is set to linear. |
| train lr warmup steps | Specifies the number of effective steps to gradually increase lr at the beginning of learning. |
| train textencoder learning rate | Learning rate of the Text Encoder. If 0, the value of train learning rate is used. |
| image buckets step | Specifies the detail of classification when classifying images into several aspect ratios. |
| image min length | Specifies the minimum resolution. |
| image max ratio | Specifies the maximum aspect ratio. |
| sub image num | The number of times the image is reduced to different resolutions. |
| image mirroring | Mirrors the image horizontally. |
| save per steps | Saves LoRA at specified steps. |
| save overwrite | Whether to overwrite when saving. |
| save as json | Whether to save the settings during learning execution. The settings are saved by date in the json folder of the extension. |
| model v pred | Whether the SD2.X model uses v-pred. |
| train model precision | Precision of non-learning targets during learning. fp16 is fine. |
| train lora precision | Precision of the learning target during learning. fp32 is fine. |
| save precision | Precision when saving. fp16 is fine. |
| train seed | The seed used during learning. |
| diff save 1st pass | Whether to save the copier LoRA. |
| diff 1st pass only | Learn only the copier LoRA. |
| diff load 1st pass | Load the copier LoRA from a file. |
| train snr gamma | Whether to add timestep correction

. Set a value between 0 and 20. The recommended value is 5. |
| logging verbose | Outputs logs to the command prompt. |
| logging_save_csv | Records step, loss, learning rate in csv format. |

## Presets, Saving and Loading of Settings
You can call up the settings with a button. The settings are handled in a json file. Presets are stored in the preset folder.

## Queue
You can reserve learning. Pressing the `Add to Queue` button reserves learning with the current settings. If you press this button during learning, the next learning will start automatically after the learning. If pressed before learning, after learning ends with the settings when `Start Training` was pressed, the learning in the Queue list is processed in order. You cannot add settings with the same `save lora name`.

## Plot
When the logging_save_csv option is enabled, you can graph the progress of learning. If you don't enter anything in `Name of logfile`, the results of learning in progress or the most recent learning are displayed. If you enter a csv file name, those results are displayed. Only the file name is needed, not the full path. The file must be in the logs folder.

## Acknowledgments
This code is based on [Plat](https://github.com/p1atdev)'s [LECO](https://github.com/p1atdev/LECO), [laksjdjf](https://github.com/laksjdjf)'s [learning code](https://github.com/laksjdjf/sd-trainer), [kohya](https://github.com/kohya-ss)'s [learning code](https://github.com/kohya-ss/sd-scripts), and [KohakuBlueleaf](https://github.com/KohakuBlueleaf)'s [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS).

## Reference
- https://github.com/rohitgandikota/erasing

- https://github.com/cloneofsimo/lora

- https://github.com/laksjdjf/sd-trainer

- https://github.com/kohya-ss/sd-scripts

- https://github.com/KohakuBlueleaf/LyCORIS

- https://github.com/ntc-ai/conceptmod
