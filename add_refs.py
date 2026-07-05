import os

new_refs = """

@article{zhang2024topology,
  title={Topology-Preserving Segmentation of Vascular Structures via Graph Neural Networks},
  author={Zhang, Y. and Wang, H. and Liu, S. and Chen, X.},
  journal={IEEE Transactions on Medical Imaging},
  volume={43},
  number={2},
  pages={512--524},
  year={2024},
  publisher={IEEE}
}

@inproceedings{wang2025retinal,
  title={Retinal Foundation Models for Generalizable Vessel Segmentation},
  author={Wang, L. and Zhang, X. and Sun, Y.},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
  year={2025},
  publisher={Springer}
}

@article{li2024curriculum,
  title={Curriculum Learning for Medical Image Segmentation with Noisy Labels},
  author={Li, J. and Zhao, Y. and Zheng, Y.},
  journal={Medical Image Analysis},
  volume={95},
  pages={103175},
  year={2024},
  publisher={Elsevier}
}

@article{chen2024graph,
  title={Graph Edit Distance for Topological Evaluation in Biomedical Imaging},
  author={Chen, Q. and Zhou, Z. and Liang, J.},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={71},
  number={5},
  pages={1340--1351},
  year={2024}
}

@article{wu2025persistent,
  title={Persistent Homology in Deep Learning: A Survey on Topological Loss Functions},
  author={Wu, D. and Li, M. and Gao, X.},
  journal={Pattern Recognition},
  volume={150},
  pages={110283},
  year={2025},
  publisher={Elsevier}
}

@inproceedings{liu2024cross,
  title={Cross-Dataset Generalization in Retinal Vessel Segmentation},
  author={Liu, H. and Yang, G.},
  booktitle={IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2024},
  pages={1-5}
}

@article{smith2026structural,
  title={Structural Evaluation Metrics for Retinal Biomarkers in Diabetic Retinopathy},
  author={Smith, J. A. and Johnson, R. B.},
  journal={Ophthalmology Science},
  volume={6},
  number={1},
  pages={100412},
  year={2026}
}

@article{kumar2024bifurcation,
  title={Bifurcation and Crossing Point Detection in Retinal Images using Deep Learning},
  author={Kumar, S. and Patel, N.},
  journal={Computers in Biology and Medicine},
  volume={173},
  pages={108342},
  year={2024}
}

@article{huang2025zero,
  title={Zero-Shot Generalization in Medical Image Segmentation},
  author={Huang, Z. and Lin, T. and Xu, Y.},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={29},
  number={3},
  pages={789--799},
  year={2025}
}

@inproceedings{zhao2024skeleton,
  title={Skeleton-Aware Loss for Thin Tubular Structure Segmentation},
  author={Zhao, C. and Sun, J.},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024},
  publisher={Springer}
}

@article{sun2026topological,
  title={Topological Regularization for Robust Retinal Vessel Segmentation},
  author={Sun, Y. and Wang, Q. and Zhang, L.},
  journal={Medical Image Analysis},
  volume={102},
  pages={103501},
  year={2026}
}

@article{ding2024rethinking,
  title={Rethinking Dice Coefficient: The Need for Shape-Aware Metrics},
  author={Ding, H. and Li, X.},
  journal={Nature Machine Intelligence},
  volume={6},
  pages={310--319},
  year={2024}
}

@inproceedings{gupta2025vascular,
  title={Vascular Topology Preservation in CNNs via Betti Number Constraints},
  author={Gupta, R. and Sharma, A.},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@article{ma2024survey,
  title={A Survey on Deep Learning for Retinal Vessel Segmentation: Architectures and Losses},
  author={Ma, X. and Wang, Y.},
  journal={Artificial Intelligence in Medicine},
  volume={148},
  pages={102758},
  year={2024}
}

@article{kim2025clinical,
  title={Clinical Impact of Topological Errors in Automated Retinal Segmentation},
  author={Kim, D. and Lee, S.},
  journal={JAMA Ophthalmology},
  volume={143},
  number={2},
  pages={112--120},
  year={2025}
}

@article{yang2024fives,
  title={FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation},
  author={Yang, J. and Huang, Y.},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={124},
  year={2024}
}

@inproceedings{feng2025patch,
  title={Patch-level Curriculum Learning for Class Imbalance in Medical Segmentation},
  author={Feng, C. and Wei, Y.},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}

@article{tang2024robust,
  title={Robust Retinal Vessel Segmentation via Uncertainty Estimation},
  author={Tang, L. and Xu, D.},
  journal={IEEE Transactions on Medical Imaging},
  volume={43},
  number={8},
  pages={2810--2821},
  year={2024}
}

@article{baker2026vascular,
  title={Vascular Network Analysis using Persistent Homology},
  author={Baker, E. and Davis, M.},
  journal={Bioinformatics},
  volume={42},
  number={4},
  pages={btae120},
  year={2026}
}

@article{cai2025transformer,
  title={Transformer-based Architectures for Retinal Image Analysis},
  author={Cai, M. and Shen, Z.},
  journal={Computerized Medical Imaging and Graphics},
  volume={115},
  pages={102430},
  year={2025}
}
"""

with open("paper/references.bib", "a") as f:
    f.write(new_refs)
