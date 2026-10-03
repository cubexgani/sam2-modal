# SAM2 Modal Deployment
Build scripts for Modal deployment and local testing.  
1. `app.py` contains a python script to run a Uvicorn server which loads the model.
2. `modal_app.py` contains the required modal deployment script.
3. `test_client.py` and `test_concurrent.py` contain backend and modal testing scripts. 

## Setup
1. Clone the repository.
2. Create a venv using `python -m venv .venv-sam2`, and activate it.
3. python -m pip install --upgrade pip setuptools wheel
4. Run these commands to install the dependencies and submodules:
```bash
# CPU-only pytorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Install sam2 submodule
git submodule update --init --recursive --depth 1 --progress
pip install -e third_party/sam2
# Install other dependencies
pip install fastapi uvicorn pydantic numpy opencv-python-headless pillow shapely rasterio pytest httpx modal
```
5. Download the model checkpoint.  
Powershell:
```pwsh
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" -OutFile "sam2.1_hiera_base_plus.pt"
```
Bash:
```bash
cd third_party/sam2/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```
6. Run `app.py` to serve the model in a local uvicorn server.
5. Run `test_client.py` and `test_concurrent.py` (don't forget to change the base url).Both files search for `test_images/nyc-test.tif` and send it to the backend, so make sure you have it. If you don't, make the `test_images` directory and put a GeoTIFF in it, and update the path in the test files accordingly.
7. To run the model in Modal, run `modal serve modal_app.py`. To deploy it, run `modal deploy modal_app.py`