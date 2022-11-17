# Size Does Matter: Size-aware Virtual Try-on via Clothing-oriented Transformation Try-on Network
## Single-garment try-on results

  - Same human tries on different clothes
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/9331_60_woH_wBanner.gif)
  - Different humans try on different clothes
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/w_60_woH_wBanner.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/m_60_woH_wBanner.gif)

## Multi-garment try-on results
  - Try-on results with tops tucked-in
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/CVPR_outfit_female_tucked_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/CVPR_outfit_male_tucked_masked.gif)
  - Try-on results without tops tucked-in
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/CVPR_outfit_female_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/CVPR_outfit_male_masked.gif)

## Multi-scale try-on results
  - Tops
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/banner.jpg)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human5_upper9_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human20_upper163_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human36_upper87_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human37_upper137_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human45_upper152_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human56_upper4_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human58_upper102_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human63_upper31_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human65_upper69_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human68_upper127_masked.gif)
  - Bottoms
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human3_lower111_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human23_lower26_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human28_lower63_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human33_lower128_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human50_lower14_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human51_lower19_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human57_lower82_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human61_lower112_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human72_lower130_masked.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/human73_lower71_masked.gif)

## Visual comparison with state-of-the-art virtual try-on methods

  - Preserving human characteristics, i.e., tattoo
    
    Due to the proposed *Clothing Elimination Policy*, COTTON is able to preserve the human characteristics, i.e. tattoo.
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/preserving%20human%20characteristics_woH.jpg)

  - Preserving clothing characteristics, i.e., neckline
    
    Our proposed *Clothing Segmentation Network* properly segments the region of clothes around the neckline that cannot be seen when people wear it. It helps COTTON to yield correct neckline type on try-on results. On the other hand, the baselines all lead to undesired noise around the neckline on the final synthesis results.
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/Vneck%20comparison_woH.jpg)
