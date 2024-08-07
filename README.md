# Local Causal Discovery with Linear non-Gaussian Cyclic Models

[Paper](https://arxiv.org/abs/2403.14843) by [Haoyue Dai](https://hyda.cc)\*, [Ignavier Ng](https://ignavierng.github.io/)\*, [Yujia Zheng](https://yjzheng.com/), [Zhengqing Gao](https://hit-chris.github.io/), [Kun Zhang](https://www.andrew.cmu.edu/user/kunz1/index.html). Appears at AISTATS 2024.

*The first to tackle local causal discovery in cyclic models. By independent subspace analysis, all the local causal structures and coefficients in the equivalence class are identified (intersecting cycles are aslo allowed). A regression-based variant is given for acyclic cases.*



## Running the Methods
- To run the method for acyclic case:
```
python acyclic_example.py
```

- To run the method for cyclic case:
```
python cyclic_example.py
```

## Acknowledgments
- The code to generate the synthetic DAGs and compute the graph metrics (e.g., SHD, TPR) is based on [NOTEARS](https://github.com/xunzheng/notears).
- The code to generate the synthetic cyclic graphs and data is based on [dglearn](https://github.com/syanga/dglearn).



## Citation

If you use this code for your research, please cite our paper:

```bibtex
@inproceedings{dai2024local,
  title={Local Causal Discovery with Linear non-Gaussian Cyclic Models}, 
  author={Haoyue Dai and Ignavier Ng and Yujia Zheng and Zhengqing Gao and Kun Zhang},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  year={2024},
  organization={PMLR}
}
```
