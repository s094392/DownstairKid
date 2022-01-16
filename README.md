# DownstairKid
![Preview](https://github.com/s094392/DownstairKid/blob/main/preview.gif)

A [DownstairKid(NS-SHAFT)](https://www.nagi-p.com/v1/fre/nsshaft.html) AI based on Deep Q-learning.

## Requirements
* wine
* Xvfb
* Pytorch

## Files
* main.py: Test the environment and make sure Xvfb works fine.
* train.py: Train the model.
* screenrecore.py Record the virtual screen.
  * set the DISPLAY environment variable to the virtual screen display id first to recorder.

## Usage
### Install Dependencies
```bash
sudo pacman -S python wine xorg-server-xvfb xdotool
pip install -r requirements.txt
# According to your environment
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### Test DownEnv
```
python main.py
```
You should see an OpenCV imshow window

### Edit Hyperparameters
train.py
```python
gamma = 0.99
batch_size = 64
replay_size = 1000
learning_rate = 1e-4
sync_target_frames = 20
replay_start_size = 1000

eps_start = 0.02
eps_decay = .999985
eps_min = 0.02
filename = "1220_s3" # Filename for best weights
```
### Train
```bash
python train.py
```
