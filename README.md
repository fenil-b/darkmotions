# DarkMotions

### Train
```
python train.py --snapshots_folder weight/ --pretrain_dir weight/Epoch99.pth
```

### Test
```
python test.py --input_dir test_dataset --weight_dir weight/Epoch99.pth --test_dir test_output
```

The stands as an modified adaption of [1] and [2] with our innovation.

[1] Zheng, Shen, and Gaurav Gupta. "Semantic-guided zero-shot learning for low-light image/video enhancement." In Proceedings of the IEEE/CVF Winter conference on applications of computer vision, pp. 581-590. 2022.

[2] A. Zhu, L. Zhang, Y. Shen, Y. Ma, S. Zhao and Y. Zhou, "Zero-Shot Restoration of Underexposed Images via Robust Retinex Decomposition," 2020 IEEE International Conference on Multimedia and Expo (ICME), London, UK, 2020, pp. 1-6, doi: 10.1109/ICME46284.2020.9102962.

[3] Wei, Chen, Wenjing Wang, Wenhan Yang, and Jiaying Liu. "Deep retinex decomposition for low-light enhancement." arXiv preprint arXiv:1808.04560 (2018).

[4] Li, Chongyi, Chunle Guo, Linghao Han, Jun Jiang, Ming-Ming Cheng, Jinwei Gu, and Chen Change Loy. "Low-light image and video enhancement using deep learning: A survey." IEEE transactions on pattern analysis and machine intelligence 44, no. 12 (2021): 9396-9416.

[5] K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning for Image Recognition," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90.
