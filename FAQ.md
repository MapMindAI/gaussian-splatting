## FAQ:

### 1. Cannot run the docker image in Ubuntu 20.04.

You need to update to Ubuntu 22.04 (at least), following:

1. Back up important files first. Ubuntu’s upgrade docs explicitly recommend backing up before a release upgrade.
2. Fully update 20.04 first

```bash
sudo apt update
sudo apt full-upgrade
sudo reboot
```

3. Start the release upgrade (https://ubuntu.com/server/docs/how-to/software/upgrade-your-release)

```bash
sudo do-release-upgrade
```

4. After install new Ubuntu, you might encounter **Display Manager Issue**. you need to install new ubuntu driver to make it working:
  * Configure network : (1) show your eth networks `ip link show`; (2) connect to eth `sudo dhclient eth0`.
  * Install nvidia drivers : `sudo ubuntu-drivers autoinstall`
  * Fix desktop env : `sudo apt install --reinstall ubuntu-desktop`

### 2. Insta360 stitch not working

You need to install cudnn following [NVIDIA official doc](https://developer.nvidia.com/cudnn-downloads).

### 3. Memory not enough


1. You could reduce the size of image by changing **--resize_factor 0.5** in "mapmind/run_360.sh".
2. To do the full training routine and avoid running out of memory, you can increase the --densify_grad_threshold (0.0002), --densification_interval (100) or reduce the value of --densify_until_iter (15_000).
