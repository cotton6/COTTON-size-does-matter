# Size Does Matter: Size-aware Virtual Try-on via Clothing-oriented Transformation Try-on Network
## Single-garment try-on results

  - Same human tries on different clothes
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/banner.jpg)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/9331_60.gif)
  - Different humans try on different clothes
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/banner.jpg)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/w_60.gif)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/banner.jpg)
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/m_60.gif)

## Visual comparison with state-of-the-art virtual try-on methods

  - Preserving human characteristics, i.e., tattoo
    
    Due to the proposed *Clothing Elimination Policy*, COTTON is able to preserve the human characteristics, i.e. tattoo.
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/preserving%20human%20characteristics.jpg)

  - Preserving clothing characteristics, i.e., neckline
    
    Our proposed *Clothing Segmentation Network* properly segments the region of clothes around the neckline that cannot be seen when people wear it. It helps COTTON to yield correct neckline type on try-on results. On the other hand, the baselines all lead to undesired noise around the neckline on the final synthesis results.
  ![image](https://github.com/cotton6/COTTON-size-does-matter/blob/main/Try-on%20results/Vneck%20comparison.jpg)
