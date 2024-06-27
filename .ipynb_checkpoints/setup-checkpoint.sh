pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115
cd hls-foundation-os
pip install -e .
pip install -U openmim
mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html
pip uninstall numpy
pip install numpy==1.26.4
pip install matplotlib
pip install huggingface_hub
pip install scipy