<div align="center">

<h1>[ICCV'23] Size Does Matter: Size-aware Virtual Try-on via Clothing-oriented Transformation Try-on Network</h1>

<div>
    Chieh-Yun Chen<sup>1,2</sup>,
    Yi-Chung Chen<sup>1,3</sup>,
    Hong-Han Shuai<sup>2</sup>,
    Wen-Huang Cheng<sup>3</sup>,
</div>
<div>
    <sup>1</sup>Stylins.ai&emsp; <sup>2</sup>National Yang Ming Chiao Tung University&emsp;  <sup>3</sup>National Taiwan University
</div>


Official Pytorch implementation [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Size_Does_Matter_Size-aware_Virtual_Try-on_via_Clothing-oriented_Transformation_Try-on_ICCV_2023_paper.pdf)][[Supplement](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Chen_Size_Does_Matter_ICCV_2023_supplemental.pdf)]

</div>

![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/TryOn_results/banner_size_noHead.jpg)

> **Abstract:** *Virtual try-on tasks aim at synthesizing realistic try-on results by trying target clothes on humans. Most previous works relied on the Thin Plate Spline or the prediction of appearance flows to warp clothes to fit human body shapes. However, both approaches cannot handle complex warping, leading to over distortion or misalignment. Furthermore, there is a critical unaddressed challenge of adjusting clothing sizes for try-on. To tackle these issues, we propose a Clothing-Oriented Transformation Try-On Network (COTTON). COTTON leverages clothing structure with landmarks and segmentation to design a novel landmark-guided transformation for precisely deforming clothes, allowing for size adjustment during try-on. Additionally, to properly remove the clothing region from the human image without losing significant human characteristics, we propose a clothing elimination policy based on both transformed clothes and human segmentation. This method enables users to try on clothes tucked-in or untucked while retaining more human characteristics. Both qualitative and quantitative results show that COTTON outperforms the state-of-the-art high-resolution virtual try-on approaches.*

## Implementation
Please see ./code for more implementation details.

## Multi-garment try-on results
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/TryOn_results/multi-garment_results_masked.gif)


## Multi-size try-on results
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/TryOn_results/human33_upper128_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/TryOn_results/human36_upper87_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/TryOn_results/human45_upper145_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/TryOn_results/human56_upper4_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/TryOn_results/human58_upper90_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/TryOn_results/human70_upper52_masked.gif)

## Visual comparison with state-of-the-art virtual try-on methods

  - Preserving human characteristics, i.e., tattoo
    
    Due to the proposed *Clothing Elimination Policy*, COTTON is able to preserve the human characteristics, i.e. tattoo.
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/TryOn_results/preserving%20human%20characteristics_woH.jpg)

  - Preserving clothing characteristics, i.e., neckline
    
    Our proposed *Clothing Segmentation Network* properly segments the region of clothes around the neckline that cannot be seen when people wear it. It helps COTTON to yield correct neckline type on try-on results. On the other hand, the baselines all lead to undesired noise around the neckline on the final synthesis results.
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/TryOn_results/Vneck%20comparison_woH.jpg)

## Citation


```bibtex
@InProceedings{Chen_2023_ICCV,
    author    = {Chen, Chieh-Yun and Chen, Yi-Chung and Shuai, Hong-Han and Cheng, Wen-Huang},
    title     = {Size Does Matter: Size-aware Virtual Try-on via Clothing-oriented Transformation Try-on Network},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {7513-7522}
}
```

