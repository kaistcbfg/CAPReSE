# CAPReSE

## Introduction
Disruption of 3D chromatin structure due to large-scale structural variation can be identified by using Hi-C technique. The utilization of the Hi-C not only enables the detection of the SV event, but also allows the interpretation of the SV's impact in terms of rearrangement of regulatory elements. To understand the cancer genome in terms of 3D genome, several Hi-C based SV detection software have been developed. As Hi-C contact maps contains signals from various sources, software that uses deep learning was also developed to successfully find SV patterns in the Hi-C contact maps. Variants of Hi-C techniques such as a single-cell Hi-C have been developed to obtain additional information other than the merged 3D chromatin structure. However, the conventional deep learning has poor classification ability for untrained data and requires a lot of data for re-training, which hinders SV detection from the new experimental technique data produced in small quantities. To solve this problem, we applied a few-shot learning method, which enables a pre-trained model to be trained using only a few labeled samples per class. The development of our method (**C**hromatin **A**nomaly **P**attern **RE**cognition and **S**ize **E**stimation, **CAPReSE**) expands the SV detection to single cell Hi-C data, which deepens our observation of the cancer 3D genome.

**CAPReSE flow diagram**
![mainflow](https://www.dropbox.com/scl/fi/1uqlgts4a71twx5mv2w3m/caprese_flow.png?rlkey=xgwkgpdbdsoctxtcwytnn0jte&dl=1)

This document also covers the usage and settings of a VM submitted to K-BDS.

## Publication and Citation
Early version of CAPReSE was published at:    
+ Kim *et al*. "Spatial and clonality-resolved 3D cancer genome alterations reveal enhancer-hijacking as a potential prognostic marker for colorectal cancer" *Cell Rep.* (2023), DOI: https://doi.org/10.1016/j.celrep.2023.112778
+ https://github.com/kaistcbfg/CAPReSEv1


## Installation configuration

+ Software version information.

Software | Version
--------- | ---------  
**VM OS**| CentOS 7.9 Minimal
**conda**| Anaconda3-2023.09-0-Linux-x86_64   
**Python**|3.10.13  
**numpy**| 1.26.2  
**scipy**| 1.11.4  
**pandas**| 2.1.3  
**opencv-python**| 4.8.1  
**mahotas**| 1.4.13   
**Pillow**| 10.0.01 
**scikit-learn**| 1.3.2  
**cooler**| 0.9.3 
**pytorch**| 2.1.1  
**CLIP**| 1.0.dist  

Version conflicts between multiple software and packages were frequently identified during installation. Please note the effect of installation order and version depending on the used platform.

+ Installation log when configuring the VM.

```bash
yum -y install git mesa-libGL # Required for OpenCV

conda create -n caprese python=3.10
conda activate caprese

pip install numpy scipy pandas opencv-python mahotas Pillow cooler scikit-learn

conda install --yes -c pytorch pytorch torchvision

pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```
## Usage

+ Pipeline was set up under **"caprese"** conda environment.
+ Codes are located under **/root/CAPReSE** directory.
+ CLIP pretrained weight is located at **/root/.cache/clip/ViT-B-32.pt**.
+ Main SV detection SW is **caprese_main.py**.
+ SW tools for train/test and feature extraction is located at **./tools**.
+ Use **--help** argument to check command options.

```bash
usage: caprese_main.py [-h] --header HEADER --input-file INPUT_FILE [--input-format INPUT_FORMAT] [--save-path SAVE_PATH] --chrom1 CHROM1 --chrom2 CHROM2 [--resolution RESOLUTION] [--clr-balance CLR_BALANCE]
                       [--imgproc-clipval IMGPROC_CLIPVAL] [--imgproc-sizefiltval IMGPROC_SIZEFILTVAL] [--imgproc-diagval IMGPROC_DIAGVAL] [--imgproc-lowcov IMGPROC_LOWCOV] [--imgproc-cropsize IMGPROC_CROPSIZE] --model-infopath
                       MODEL_INFOPATH --model-ptpath MODEL_PTPATH [--fdist-cutoff FDIST_CUTOFF] [--visualize-flag VISUALIZE_FLAG] [--visualize-outdir VISUALIZE_OUTDIR] [--visualize-boxsize VISUALIZE_BOXSIZE]
                       [--imgproc-getcrop IMGPROC_GETCROP] [--version]

CAPReSE: Chromatin Anomaly Pattern REcognition and Size Estimation

optional arguments:
  -h, --help            show this help message and exit
  --header HEADER       save file header
  --input-file INPUT_FILE
                        input *.pkl.gz, *.cool, or *.mcool file
  --input-format INPUT_FORMAT
                        default pickle, format: pickle (numpy array pickle) or mccol
  --save-path SAVE_PATH
  --chrom1 CHROM1       taget chr name
  --chrom2 CHROM2       if chrom1==chrom2: cis-
  --resolution RESOLUTION
                        bin resolution (default 40kb)
  --clr-balance CLR_BALANCE
                        cooler balance option (default False)
  --imgproc-clipval IMGPROC_CLIPVAL
                        Hi-C contact map 255 scale conversion cutoff value (default 5)
  --imgproc-sizefiltval IMGPROC_SIZEFILTVAL
                        Contour size filter cutoff (default 2)
  --imgproc-diagval IMGPROC_DIAGVAL
                        Distance from diagonal (default 800000)
  --imgproc-lowcov IMGPROC_LOWCOV
                        0-1 Low coverage ratio cutoff (default 0.3)
  --imgproc-cropsize IMGPROC_CROPSIZE
                        Half of boxsize of breakpoint centered crop(default 16)
  --model-infopath MODEL_INFOPATH
                        PATH to model info pkl.gz file
  --model-ptpath MODEL_PTPATH
                        PATH to pytorch model pt file
  --fdist-cutoff FDIST_CUTOFF
                        Euclidean distance from train set (default 3)
  --visualize-flag VISUALIZE_FLAG
                        Save detection result to png
  --visualize-outdir VISUALIZE_OUTDIR
                        png file save path
  --visualize-boxsize VISUALIZE_BOXSIZE
                        Half of boxsize of breakpoint centered crop(default 16
  --imgproc-getcrop IMGPROC_GETCROP
                        Collect crop without clf
  --version             show program's version number and exit
```
+ **--imgproc-getcrop** flag will collect image crops without classification, saved under 'crop' directory.
+ Parameters are optimized for 40kb [covNorm](https://github.com/kaistcbfg/covNormRpkg) normalized Hi-C contact maps. Change parameters based on the resolution, depth, or sample condition.

## Example

+ Example data are located under **./data/example** directory.
+ 6 CLIP features (pos/neg for each base, train, and test), pkl.gz file, and mcool example files are given.
+ Use **./test_run.sh** script to run example cases.
+ **./test_run.sh** will generate two csv CLIP feature files, model.pt file, and info.pkl.gz file. Crops will be collected under result directory.
+ Note that the generated data are for the demonstration. **Fine tune with proper dataset and parameter for actual application.**

**Scripts in ./tools example flow diagram**
![toolflow](https://www.dropbox.com/scl/fi/lzjim6bxxza7p55f66ikh/caprese_tool_flow.png?rlkey=9xyj2hqc2cfqkvhzoh5cflttl&dl=1)

**test_run.sh code**
```bash
# Crop collection
python caprese_main.py --header K562_example --input-file ./data/example/K562_chr9.chr22.bulk.pkl.gz --input-format pickle --chrom1 chr9 --chrom2 chr22 --model-infopath ./TipAdapterF_info.pkl.gz  --model-ptpath ./TipAdapterF_model.pt --imgproc-getcrop True

# CLIP feature generation
python ./tools/imgdir_CLIPfeature_aug.py --input-dir ./data/example/crop/pos --num-aug 3 --output-filename ./CLIPfeatures_pos.csv
python ./tools/imgdir_CLIPfeature_aug.py --input-dir ./data/example/crop/neg --num-aug 3 --output-filename ./CLIPfeatures_neg.csv

# Train-Test
python ./tools/train_tip_adapter_F.py --input-poswfile ./data/example/CLIPfeature/weight_pos.csv --input-negwfile ./data/example/CLIPfeature/weight_neg.csv  --input-postfile ./data/example/CLIPfeature/train_pos.csv  --input-negtfile ./data/example/CLIPfeature/train_neg.csv
python ./tools/test_tip_adapter_F.py --model-ptpath ./TipAdapterF_model.pt  --model-infopath ./TipAdapterF_info.pkl.gz --input-postest ./data/example/CLIPfeature/test_pos.csv  --input-negtest ./data/example/CLIPfeature/test_neg.csv

# Classification
python caprese_main.py --header K562_example --input-file ./data/example/K562_chr9.chr22.bulk.pkl.gz --input-format pickle --chrom1 chr9 --chrom2 chr22 --model-infopath ./TipAdapterF_info.pkl.gz  --model-ptpath ./TipAdapterF_model.pt
python caprese_main.py --header K562_example --input-file ./data/example/K562.mcool --input-format mcool --chrom1 chr9 --chrom2 chr22 --model-infopath ./TipAdapterF_info.pkl.gz  --model-ptpath ./TipAdapterF_model.pt
```

+ Train/Test script example output

```bash
# Train phase log
Train base weight loaded
pos: ./data/example/CLIPfeature/weight_pos.csv 328
neg: ./data/example/CLIPfeature/weight_neg.csv 221
tot: 549 x 512

Train data loaded
pos: ./data/example/CLIPfeature/train_pos.csv 218
neg: ./data/example/CLIPfeature/train_neg.csv 639

Train options
epoch: 25 | batch size: 5 | beta: 1.0
Train log
epoch: 0 | loss: 82.49039954692125
epoch: 1 | loss: 53.202773278579116
epoch: 2 | loss: 43.350863073486835
epoch: 3 | loss: 38.20283195050433
epoch: 4 | loss: 35.08020553085953
epoch: 5 | loss: 36.34667465909479
epoch: 6 | loss: 30.226710107177496
epoch: 7 | loss: 29.586680495929613
epoch: 8 | loss: 29.973209175048396
epoch: 9 | loss: 28.593080126214772
epoch: 10 | loss: 26.776456548657734
epoch: 11 | loss: 25.12640765949618
epoch: 12 | loss: 28.83041144438903
epoch: 13 | loss: 22.53082690143492
epoch: 14 | loss: 22.138663919758983
epoch: 15 | loss: 19.813909954711562
epoch: 16 | loss: 20.272727015486453
epoch: 17 | loss: 22.330062292472576
epoch: 18 | loss: 19.470869253054843
epoch: 19 | loss: 21.52079430595404
epoch: 20 | loss: 16.720515411805536
epoch: 21 | loss: 16.093178928068482
epoch: 22 | loss: 16.320438225164253
epoch: 23 | loss: 14.464738617196417
epoch: 24 | loss: 14.372195415411625

Model pt file saved as ./TipAdapterF_model.pt
Model info pkl saved as ./TipAdapterF_info.pkl.gz

#Test phase log
Model info loaded. ./TipAdapterF_info.pkl.gz
Model loaded. ./TipAdapterF_model.pt
Precision: 0.94 | Recall: 0.71
Neg True: 726 | Pos True: 337
Neg corr: 710 | Pos corr: 240
```