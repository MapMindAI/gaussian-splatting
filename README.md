
# Readme for MindMapAI

# 🚀 Getting Started
## 1. Prepare the environment

We recommend using the provided Docker image to ensure a consistent environment.
`docker pull ghcr.io/mapmindai/myapp:sha-881caec`

run the container with:
```
docker run -it --rm -v $(pwd):/workspace ghcr.io/MapMindAI/myapp:latest
```

<details>
<summary>Or build the submodules if not using docker image</summary>
in the repo we have rebuilt libs for docker env, if you don't use docker, you might need to build these libs:
```
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim
```
</details>

## 2. Run With Drone Data

1. Put the data to folder ([example google drive drone videos](https://drive.google.com/drive/folders/1TIcNHhN6kdgpAfCDT56L06swd2MmmnuI?usp=drive_link)
):
  * Put the drone video to the session_folder.
  * If you want to build with images, create a folder called "images", and put you photos there.

![example folder structure](assets/mapmind/example_drone_data.png)

2. Run the script:

```
./mindmap/run_drone.sh MAP_FOLDER SESSION_NAME
```

Example usage : `./mindmap/run_drone.sh /mnt/data/yeliu/gaussian_splatting DJI_test`. After the building step finished, we will have the following results in the folder, and gaussian splatting point cloud could be found in 'output' folder:

![example folder result](assets/mapmind/example_drone_data_result.png)


## 3. Visualization

* 👑 use https://playcanvas.com/supersplat/editor
* 👍 using the threejs version from https://discourse.threejs.org/t/3d-gaussian-splatting-in-three-js/57858 in https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php
