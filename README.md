# Mitigating Biases in Blackbox Feature Extractors for Image Classification Tasks

This repository contains the implementation of our margin-loss based model.

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

* Download the CelebA, and Waterbirds datasets. Extract their features from a pretrained feature extractor and store them. Note the folder path and update it in `utils/config.py`. Recall that the validation set has to be group-balanced. For CelebA, store the folder in the `celeba_path` attribute of `utils/config.py` (similarly for Waterbirds). The embeddings should be stored in `{split}_feats.npy`, whereas the bias and target labels should be stored in `{split}_bias.npy` and `{split}_target.npy` respectively, inside the folder. 

</br>

## Usage instructions:

### Baseline
To run the baseline (ERM model), run the following command. Dataset can either be waterbirds or celeba:

```bash
python margin_loss.py --dataset waterbirds --train --type baseline --bias 
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
Checkpoints of ERM and our model are provided in [erm_path](./basemodel.pth) and [margin_path](./margin.pth). These can be used to validate the scores reported in our paper.

</br>

## Results on our method evaluated on Waterbirds and CelebA on the ResNet-18 backbone

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
    <td class="tg-c3ow">38.90</td>
    <td class="tg-c3ow">76.22</td>
    <td class="tg-c3ow">27.20</td>
    <td class="tg-c3ow">75.43</td>
  </tr>
  <tr>
    <td class="tg-0pky">DebiAN</td>
    <td class="tg-c3ow">58.94</td>
    <td class="tg-c3ow">80.47</td>
    <td class="tg-c3ow">26.10</td>
    <td class="tg-c3ow">75.41</td>
  </tr>
  <tr>
    <td class="tg-0pky">BPA</td>
    <td class="tg-c3ow">58.70</td>
    <td class="tg-c3ow">80.83</td>
    <td class="tg-c3ow">66.71</td>
    <td class="tg-c3ow">84.14</td>
  </tr>
  <tr>
    <td class="tg-0pky">LfF</td>
    <td class="tg-c3ow">66.09</td>
    <td class="tg-c3ow">81.39</td>
    <td class="tg-c3ow">13.26</td>
    <td class="tg-c3ow">69.42</td>
  </tr>
  <tr>
    <td class="tg-0pky">JTT</td>
    <td class="tg-c3ow">49.84</td>
    <td class="tg-c3ow">77.03</td>
    <td class="tg-c3ow">56.25</td>
    <td class="tg-c3ow">73.58</td>
  </tr>
  <tr>
    <td class="tg-0pky">GEORGE</td>
    <td class="tg-c3ow">59.35</td>
    <td class="tg-c3ow">80.34</td>
    <td class="tg-c3ow">42.22</td>
    <td class="tg-c3ow">79.76</td>
  </tr>
  <tr>
    <td class="tg-0pky">Ours</td>
    <td class="tg-c3ow">80.29</td>
    <td class="tg-c3ow">84.56</td>
    <td class="tg-c3ow">81.61</td>
    <td class="tg-c3ow">86.04</td>
  </tr>
</tbody>
</table>