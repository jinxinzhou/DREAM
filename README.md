<p align="center">

  <h2 align="center"><b>DREAM</b>: Diffusion Rectification and Estimation-Adaptive Models</h2>
  <p align="center">
    <a style="text-decoration:none" href="https://scholar.google.com/citations?user=XR5CQJcAAAAJ">
                       Jinxin Zhou</a><sup>1,*</sup>
    &nbsp;&nbsp;
    <a style="text-decoration:none" href="https://www.tianyuding.com/">
                        Tianyu Ding</a><sup>2,*,&dagger;</sup>
    &nbsp;&nbsp;
    <a style="text-decoration:none" href="https://scholar.google.com/citations?user=2BahjdkAAAAJ&hl=en">
                       Tianyi Chen</a><sup>2</sup>
    &nbsp;&nbsp;
    <a style="text-decoration:none" href="http://www.jiachenjiang.com/">
                    Jiachen Jiang</a><sup>2</sup>
    &nbsp;&nbsp;
    <a style="text-decoration:none" href="https://www.microsoft.com/applied-sciences/people/ilya-zharkov">
                    Ilya Zharkov</a><sup>2</sup>
    &nbsp;&nbsp;
    <a style="text-decoration:none" href="https://zhihuizhu.github.io">
                     Zhihui Zhu</a><sup>1</sup>
    &nbsp;&nbsp;
    <a style="text-decoration:none" href="https://sites.google.com/site/lumingliangshomepage/">
                     Luming Liang</a><sup>2,&dagger;</sup>
    <br>
    <sup>1</sup>Ohio State University &nbsp;&nbsp;&nbsp; <sup>2</sup>Microsoft
    <br>
    </br>
  <a href="https://www.tianyuding.com/project-pages/DREAM/"><strong>Project Page</strong></a> | <a href="https://www.tianyuding.com/project-pages/DREAM/"><strong>Paper</strong></a>
  </p>
</p>
<div align="center">
  <br>
  <img src="./teaser.png" alt="<p class="text-center" style="padding-top: 15px; margin-bottom: -3px;">Turning the top to the bottom by adding only three lines of code.</p>
</div>

We present **DREAM**, a novel training framework representing **D**iffusion **R**ectification and **E**stimation-**A**daptive **M**odels, requiring minimal code changes (just three lines) yet significantly enhancing the alignment of training with sampling in diffusion models. DREAM features two components: diffusion rectification, which adjusts training to reflect the sampling process, and estimation adaptation, which balances perception against distortion. When applied to image super-resolution (SR), DREAM adeptly navigates the tradeoff between minimizing distortion and preserving high image quality. Experiments demonstrate DREAM's superiority over standard diffusion-based SR methods, showing a 2 to 3x faster training convergence and a 10 to 20x reduction in necessary sampling steps to achieve comparable or superior results. We hope DREAM will inspire a rethinking of diffusion model training paradigms.

## Citation
If you find our work helpful, please kindly cite our work:
```BibTeX
@article{TODO,
    author = {TODO},
    title  = {TODO},
    joural = {ArXiv},
    year   = {2023},
}
```
