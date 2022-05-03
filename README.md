# SVA
Black-box video attack；Video attack with reinforcement learning；SVA

Code for paper " [Sparse Black-box Video Attack with Reinforcement](https://arxiv.org/abs/2001.03754)"

Here, we provide some video clips in folder "TT", you can run the following script and observe the effect:

```python
python un_attack_show.py   # untargeted attack
```

You can also test targeted attacks by changing the attack function.


**Threat models:**

The video classification model, please refer to project https://github.com/FenHua/action-recognition

You can train the recognition model with your own data, and use the video attack method to attack them.



**The complete project will be updated soon.**

If you find some contents useful for your research, please cite:

```
@article{yan2020sparse,
  title={Sparse black-box video attack with reinforcement learning},
  author={Wei, Xingxing and Yan, Huanqian and Li, Bo},
  journal={arXiv preprint arXiv:2001.03754},
  year={2020}
}
```
