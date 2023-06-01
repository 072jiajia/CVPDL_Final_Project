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

COCO
```
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

## Prepare Images' Embedding
It takes about 30 and 60 minutes for vit_b and vit_h
```
    CUDA_VISIBLE_DEVICES={GPU_ID} sh scripts/preprocess.sh vit_h
```
COCO:
```
    sh scripts/preprocess_coco.sh vit_h
```

## Do Experiments
To do an experiment on the naive prompt learning method, use `--dataset` to specify the type of dataset (`VOC` or `COCO`). 
```
    CUDA_VISIBLE_DEVICES={GPU_ID} python prompt_learning.py --n-emb 1
```

I notice that using multiple tokens for one class can boost the performace
```
    CUDA_VISIBLE_DEVICES={GPU_ID} python prompt_learning.py --n-emb 2
```

CoCoOp
```
    CUDA_VISIBLE_DEVICES={GPU_ID} python prompt_learning.py --n-emb {N} --trainer cocoop --batch-size 8
```

# Experimental Results
Fully supervised method (CoOp, epoch 200)

| number of tokens | 1 | 2 | 3 | 4 | 5 | 6 |
|:-: |:-: |:-: |:-: |:-: |:-: |:-: |
|mIoU| 0.645413| 0.667269| 0.694688| 0.709082|  0.720209| 0.719938|


Fully supervised method (CoCoOp, epoch 200)

| number of tokens | 1 | 2 | 3 | 4 | 5 | 6 |
|:-: |:-: |:-: |:-: |:-: |:-: |:-: |
|mIoU| 0.743866| | 0.773365| 0.760778| 0.767652| 0.777974|


Zero Shot method (CoOp)
||mIoU| aeroplane | bicycle| bird| boat| bottle| bus| car| cat| chair| cow| diningtable | dog| horse| motorbike| person|pottedplant| sheep| sofa| train| tvmonitor |
|:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |
|Training | | v| v| v| v| v| v| v| v| v| v| v| v| v| v| v| v| | | | |
n-emb = 1| 0.6077 |  0.9048| 0.7908| 0.3244| 0.6899| 0.5939| 0.3517| 0.8240| 0.4982| 0.8356| 0.1126| 0.7978| 0.3493| 0.7815| 0.7965| 0.8192| 0.6871| 0.1100| 0.7352| 0.2013| 0.7659|
n-emb = 2| 0.6092 |  0.9099| 0.8631| 0.2345| 0.6774| 0.6188| 0.3348| 0.8160| 0.4550| 0.8551| 0.1169| 0.7768| 0.3109| 0.7637| 0.7607| 0.8338| 0.7724| 0.1262| 0.7168| 0.2016| 0.8329|
n-emb = 3| 0.6580 |  0.9149| 0.8567| 0.3870| 0.6858| 0.6284| 0.4393| 0.8799| 0.5200| 0.8565| 0.2788| 0.8315| 0.3751| 0.8075| 0.8174| 0.8405| 0.7768| 0.2135| 0.8341| 0.2410| 0.7881|
n-emb = 4| 0.6580 |  0.9172| 0.8668| 0.3628| 0.6692| 0.6090| 0.4379| 0.8449| 0.4987| 0.8550| 0.2918| 0.8063| 0.3581| 0.8324| 0.7880| 0.8481| 0.8326| 0.2590| 0.7481| 0.2827| 0.8054|
n-emb = 5| 0.6901 |  0.9206| 0.8757| 0.4020| 0.7563| 0.6608| 0.5710| 0.8662| 0.5047| 0.8696| 0.2938| 0.8526| 0.4193| 0.8377| 0.8117| 0.8491| 0.8228| 0.2709| 0.7967| 0.2404| 0.8207|
n-emb = 6| 0.6668 |  0.9203| 0.8741| 0.3664| 0.7279| 0.6711| 0.5304| 0.8498| 0.5276| 0.8527| 0.2912| 0.8310| 0.4178| 0.7725| 0.8260| 0.8106| 0.8030| 0.2332| 0.7840| 0.1972| 0.8412|


Zero Shot method (CoCoOp)
||mIoU| aeroplane | bicycle| bird| boat| bottle| bus| car| cat| chair| cow| diningtable | dog| horse| motorbike| person|pottedplant| sheep| sofa| train| tvmonitor |
|:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |:-: |
|Training | | v| v| v| v| v| v| v| v| v| v| v| v| v| v| v| v| | | | |
n-emb = 1| 0.6887| 0.8361| 0.4244| 0.8126| 0.6920| 0.6226| 0.8795| 0.6219| 0.8866| 0.3552| 0.8924| 0.5344| 0.8652| 0.8271| 0.8496| 0.8194| 0.4138| 0.7113| 0.3127| 0.2585| 0.1917|
n-emb = 2| 0.6921| 0.8940| 0.3814| 0.8349| 0.6092| 0.7026| 0.8834| 0.6693| 0.9027| 0.3733| 0.8864| 0.5108| 0.8529| 0.8444| 0.8478| 0.8188| 0.4477| 0.8207| 0.3497| 0.1865| 0.1870|
n-emb = 3| 0.7148| 0.8829| 0.3752| 0.8074| 0.6451| 0.7043| 0.8587| 0.6977| 0.8895| 0.3827| 0.8642| 0.5382| 0.8555| 0.8288| 0.8392| 0.8256| 0.4136| 0.8148| 0.3720| 0.5255| 0.1750|
n-emb = 4| 0.7012| 0.9173| 0.4796| 0.8157| 0.7270| 0.6544| 0.8859| 0.6651| 0.8891| 0.3749| 0.8906| 0.5860| 0.8433| 0.8549| 0.8721| 0.8574| 0.4524| 0.5913| 0.2907| 0.2599| 0.3147|
n-emb = 5| 0.6984| 0.9019| 0.4401| 0.8611| 0.7298| 0.6798| 0.8854| 0.6727| 0.8817| 0.4084| 0.8567| 0.5396| 0.8614| 0.8370| 0.8583| 0.8328| 0.4093| 0.7218| 0.3297| 0.2027| 0.2045|
n-emb = 6| 0.7144| 0.9141| 0.4752| 0.8379| 0.7197| 0.6670| 0.9090| 0.6820| 0.8696| 0.4000| 0.8838| 0.5985| 0.8347| 0.8380| 0.8595| 0.8398| 0.4747| 0.7726| 0.2811| 0.3608| 0.2198|


# To Do

- [x] Add Seen/Unseen splits to VOC2012Dataset
    - [x] Train on class -[1~16] ("aeroplane" ~ "pottedplant")
    - [x] Test on class -[17~20] ("sheep", "sofa", "train", "tvmonitor")
- [x] Add checkpoint saver
- [x] Add logger 
- [x] Record experimental results
- [ ] Rewrite the code according to your familiar coding style
- [ ] Add explicit data types to the arguments of functions
- [ ] Add MSCOCO
