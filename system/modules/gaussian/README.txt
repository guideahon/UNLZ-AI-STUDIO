# SharpSplat

This software allows you to create 3D Gaussian Splats from photos and view them on your Meta Quest 3.

## Prerequisites
- Windows 10/11
- NVIDIA GPU (Recommended for faster processing)
- Meta Quest 3

## Installation
1.  Double-click `setup.bat`. This will create the environment and install dependencies. You only need to do this once.

## Usage
1.  Double-click `run_splat.bat`.
2.  Enter the path to your image (or a folder of images) when prompted. You can drag and drop the file/folder into the window.
3.  Wait for the processing to finish.
4.  The output folder will open automatically.

## Viewing on Quest 3
1.  Connect your Quest 3 to your PC via USB.
2.  Allow file access in the headset.
3.  Copy the generated `gaussians` folder (inside `output\splat_...`) to your Quest 3 (e.g., to the `Download` folder).
4.  **Option A (Native Browser):** Open the Meta Browser on Quest 3, navigate to `file:///sdcard/Download/gaussians/index.html` (path may vary slightly).
5.  **Option B (Scaniverse):** Open "Into the Scaniverse" app and import the `scene.ply` file.

## Troubleshooting
- If `setup.bat` fails, ensure you have Python installed and added to PATH.
- If processing is slow, ensure you are using a GPU.
