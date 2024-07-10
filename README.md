# Mitigation of Unknown Biases from Blackbox Feature Extractors in Image Classification Tasks

This repository contains the implementation of our model CAML.

Almost all flags can be set at `utils/config.py`. The dataset paths, the hyperparams can be set accordingly in this 
file.

## Recommended hardware configuration: 
-   GPU: One NVIDIA GeForce RTX 2080 Ti
-   RAM: 4GB (approx) if the features are explicitly extracted inside the model, 1.5GB otherwise
-   CPU: AMD Ryzen Threadripper 3960X 24-Core Processor
-   OS: Ubuntu 18.04.6 LTS
	
</br>
We used 3 random seeds to run the final models: 2411, 5193, 4594

</br>

## Dataset

* Download the CelebA, Waterbirds and ColorMNIST datasets. Extract their features from a pretrained feature extractor and store them. Note the folder path and update it in `dataset.py`

</br>

## Usage instructions:

### Baseline
To run the baseline (ERM model), run the following command. Dataset can either be waterbirds or celeba:

```bash
python margin_loss.py --dataset waterbirds --train --type baseline 
```

### Margin loss (CAML) training
```bash
python margin_loss.py --dataset waterbirds --train --type margin
```

### Model evaluation instruction
```bash
python margin_loss.py --dataset waterbirds --val-only [or test-only]
```

### Calculate NMI
```bash
python margin_loss.py --dataset waterbirds --clustering
```

### Eval only
Checkpoints of ERM and our model are provided in [erm_path](./basemodel.pth) and [caml_path](./margin.pth). These can be used to validate the scores reported in our paper.

</br>

## Results on CAML evaluated on Waterbirds and CelebA

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" rowspan="2">Model</th>
    <th class="tg-c3ow" colspan="2">Waterbirds</th>
    <th class="tg-c3ow" colspan="2">CelebA</th>
  </tr>
  <tr>
    <th class="tg-c3ow">Worst</th>
    <th class="tg-c3ow">Average</th>
    <th class="tg-c3ow">Worst</th>
    <th class="tg-c3ow">Average</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">ERM</td>
    <td class="tg-c3ow">40.71</td>
    <td class="tg-c3ow">75.55</td>
    <td class="tg-c3ow">22.22</td>
    <td class="tg-c3ow">73.93</td>
  </tr>
  <tr>
    <td class="tg-0pky">DebiAN</td>
    <td class="tg-c3ow">42.92</td>
    <td class="tg-c3ow">75.49</td>
    <td class="tg-c3ow">38.31</td>
    <td class="tg-c3ow">78.63</td>
  </tr>
  <tr>
    <td class="tg-0pky">BPA</td>
    <td class="tg-c3ow">50.03</td>
    <td class="tg-c3ow">77.08</td>
    <td class="tg-c3ow">76.12</td>
    <td class="tg-c3ow">85.09</td>
  </tr>
  <tr>
    <td class="tg-0pky">BPA*</td>
    <td class="tg-c3ow">28.13</td>
    <td class="tg-c3ow">70.65</td>
    <td class="tg-c3ow">67.21</td>
    <td class="tg-c3ow">84.62</td>
  </tr>
  <tr>
    <td class="tg-0pky">LfF</td>
    <td class="tg-c3ow">75.83</td>
    <td class="tg-c3ow">76.88</td>
    <td class="tg-c3ow">42.69</td>
    <td class="tg-c3ow">75.11</td>
  </tr>
  <tr>
    <td class="tg-0pky">JTT</td>
    <td class="tg-c3ow">1.56</td>
    <td class="tg-c3ow">59.08</td>
    <td class="tg-c3ow">68.59</td>
    <td class="tg-c3ow">80.39</td>
  </tr>
  <tr>
    <td class="tg-0pky">GEORGE</td>
    <td class="tg-c3ow">35.52</td>
    <td class="tg-c3ow">72.18</td>
    <td class="tg-c3ow">41.74</td>
    <td class="tg-c3ow">71.12</td>
  </tr>
  <tr>
    <td class="tg-0pky">GEORGE*</td>
    <td class="tg-c3ow">49.53</td>
    <td class="tg-c3ow">70.56</td>
    <td class="tg-c3ow">72.22</td>
    <td class="tg-c3ow">82.26</td>
  </tr>
  <tr>
    <td class="tg-0pky">CAML (Ours)</td>
    <td class="tg-c3ow">71.77</td>
    <td class="tg-c3ow">80.48</td>
    <td class="tg-c3ow">78.53</td>
    <td class="tg-c3ow">85.09</td>
  </tr>
  <tr>
    <td class="tg-0pky">Group-DRO</td>
    <td class="tg-c3ow">84.74</td>
    <td class="tg-c3ow">86.78</td>
    <td class="tg-c3ow">79.99</td>
    <td class="tg-c3ow">82.46</td>
  </tr>
</tbody>
</table>