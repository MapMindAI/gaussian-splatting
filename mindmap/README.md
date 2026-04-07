
# Readme for deepmirror

```
ln -s /mnt/data/yeliu/ /run/determined/workdir

ln -s /mnt/data/wenhao /run/determined/workdir
ln -s /mnt/data/yeliu/Dev/GaussianSplatting /run/determined/workdir
ln -s /mnt/data/yeliu/gaussian_splatting /run/determined/workdir
```

## add submodules

use the prebuilt version
```
export PYTHONPATH="$PYTHONPATH:/GaussianSplatting/submodules/libs"
```

<details>
<summary>Or build the submodules</summary>

```
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
```
</details>

## train

```
SESSION=wuxi
python train.py --source_path ./data/${SESSION}/dense --iterations 30_000 \
--position_lr_init 0.000016 --scaling_lr 0.001
```

<details>
<summary>Style Transfer</summary>

using https://github.com/pytorch/examples/tree/main/fast_neural_style : "candy.pth  mosaic.pth  rain_princess.pth  udnie.pth"

**Train with coco images:**
```
cd /mnt/data/yeliu/gaussian_splatting/GaussianSplatting

MODEL_PATH=/mnt/data/yeliu/models/style_models
python dm/style_transfer/neural_style.py train \
--dataset /mnt/data/caizebin/magicpoint/mg_coco_v2s_2/images \
--style-image ${MODEL_PATH}/winter/winter.png \
--save-model-dir ${MODEL_PATH}/winter/winter \
--content-weight 1e5 --style-weight 5e10 \
--epochs 3 --cuda 1
```

</details>

## test with determined ai

* **copy the data to ml data**
```
ssh yeliu@cpu001.corp.deepmirror.com

mkdir /mnt/ml-experiment-data/yeliu/gaussian_splatting/${SESSION}
cp -R /mnt/gz01/experiment/mobili/reconstruction/${SESSION}/dense /mnt/ml-experiment-data/yeliu/gaussian_splatting/${SESSION}
chmod -R 777 /mnt/ml-experiment-data/yeliu/gaussian_splatting/${SESSION}
```
* **basic run** `./dm/run_session.sh ${SESSION}`
* **process depth** `./dm/run_session.sh ${SESSION} depth`
* **process segmentation** `./dm/run_session.sh ${SESSION} segment`
* **process style** `./dm/run_session.sh ${SESSION} style your_style`

## visualization

* 👑 use https://playcanvas.com/supersplat/editor
* 👍 using the threejs version from https://discourse.threejs.org/t/3d-gaussian-splatting-in-three-js/57858 in https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php

https://github.com/user-attachments/assets/926d8789-e6ab-40ca-aafc-635c9d37f88a


## Sugar Test

[SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering](https://github.com/Anttwo/SuGaR)

```bash
SESSION=wuxi_12_day_full
python train_full_pipeline.py \
-s /mnt/data/yeliu/gaussian_splatting/${SESSION}/dense \
--regularization_type dn_consistency \
--high_poly True --export_obj True --eval False --white_background True \
--gs_output_dir /mnt/data/yeliu/gaussian_splatting/${SESSION}/output_seg
```
