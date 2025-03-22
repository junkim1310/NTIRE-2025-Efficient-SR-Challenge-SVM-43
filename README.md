# SepSRNet (Team SVM) — NTIRE 2025 Challenge on Efficient Super-Resolution @ [CVPR 2025](https://cvpr.thecvf.com/)

Official submission by **Team SVM** for the [NTIRE 2025 Challenge on Efficient Super-Resolution](https://cvlai.net/ntire/2025/).

🔗 Team page: [https://sites.google.com/view/csi2267svm/](https://sites.google.com/view/csi2267svm/)


<div align=center>
<img src="https://github.com/junkim1310/NTIRE-2025-Efficient-SR-Challenge-SVM-43/blob/main/figs/Overall_architecture.jpg" width="1000px"/> 
</div>

## The Environments

The evaluation environments adopted by us is recorded in the `requirements.txt`. After you built your own basic Python (Python = 3.9 in our setting) setup via either *virtual environment* or *anaconda*, please try to keep similar to it via:

- Step1: install Pytorch first:
`pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117`

- Step2: install other libs via:
```pip install -r requirements.txt```

or take it as a reference based on your original environments.

## How to test the baseline model?

1. `git clone https://github.com/junkim1310/NTIRE-2025-Efficient-SR-Challenge-SVM-43.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 43
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.
3. More detailed example-command can be found in `run.sh` for your convenience.

We provide the performance of our SepSRNet measured on an NVIDIA RTX A6000 GPU as follows:
- Average PSNR on DIV2K_LSDIR_valid: 26.92 dB
- Number of parameters: 0.251 M
- Number of parameters: 0.276 M
- Runtime: 25.53 ms on DIV2K_LSDIR_valid data
- FLOPs on an LR image of size 256×256: 13.39 G

## Where are the model and weight files?

- The model architecture is defined in `models/team43_SepSRNet.py`.
- The pre-trained weights are provided in `model_zoo/team43_SepSRNet.pth`.

## Participants
- Jun Young Kim (junkim1310@dgu.ac.kr)  
- Jae Hyeon Park (pjh0011@dongguk.edu)  
- Bo Gyeong Kim (qhruddl51@dgu.ac.kr)  
- Sung In Cho (csi2267@dongguk.edu)

If you have any question, feel free to reach out the contact persons and direct managers of the NTIRE challenge.

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
