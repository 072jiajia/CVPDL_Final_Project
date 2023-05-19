# AI Final Project

## Prepare Environment
```
    conda create -n cvpdl_final python=3.10
    conda activate cvpdl_final
    pip install -r requirements.txt
    pip install git+https://github.com/openai/CLIP.git
```

## Prepare Dataset
```
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    tar xvf VOCtrainval_11-May-2012.tar
    mv VOCdevkit/VOC2012 .
    rm -r VOCdevkit/
```

## Prepare Images' Embedding
It takes about 30 and 60 minutes for vit_b and vit_h
```
    CUDA_VISIBLE_DEVICES={GPU_ID} sh scripts/preprocess.sh vit_h
```

## Do Experiments
To do an experiment on the naive prompt learning method
```
    CUDA_VISIBLE_DEVICES={GPU_ID} python prompt_learning.py --n-emb 1
```

I notice that using multiple tokens for one class can boost the performace
```
    CUDA_VISIBLE_DEVICES={GPU_ID} python prompt_learning.py --n-emb 2
```

# Experimental Results
Fully supervised method


epoch 200
| number of tokens | 1 | 2 | 3 | 4 | 5 | ... |
|-|-|-|-|-|-|-|
|mIoU| 0.645413| 0.667269| 0.694688| 0.709082|  0.720209| ...|

Zero Shot method
|classes| aeroplane | bicycle| bird| boat| bottle| bus| car| cat| chair| cow| diningtable | dog| horse| motorbike| person|pottedplant| sheep| sofa| train| tvmonitor |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|Training | v| v| v| v| v| v| v| v| v| v| v| v| v| v| v| v| | | | |
|Test mIoU | TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|TBD|



# To Do

- [ ] Add Seen/Unseen splits to VOC2012Dataset
    - [ ] Train on class -[1~16] ("aeroplane" ~ "pottedplant")
    - [ ] Test on class -[17~20] ("sheep", "sofa", "train", "tvmonitor")
- [x] Add checkpoint saver
- [ ] Add logger 
- [ ] Record experimental results
- [ ] Rewrite the code according to your familiar coding style
- [ ] Add explicit data types to the arguments of functions
- [ ] Add MSCOCO
