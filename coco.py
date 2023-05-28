# for finding COCO API
# No actual use

from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

dataDir='./COCO2017'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

coco=COCO(annFile)
all_catIDs = coco.getCatIds()
all_cats = coco.loadCats(all_catIDs)

for i in all_catIDs:
    imgIds = coco.getImgIds(catIds=i)
    print(f"Number of images containing all the {i} classes:", len(imgIds))
raise InterruptedError

filterClasses = ['laptop', 'tv', 'cell phone']

catIds = coco.getCatIds(catNms=filterClasses)
imgIds = coco.getImgIds(catIds=catIds)

def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return "None"

print("Number of images containing all the  classes:", len(imgIds))

# load and display a random image
img = coco.loadImgs(imgIds[0])[0]
# print(img)
# {'license': 2, 
#  'file_name': '000000160556.jpg', 
#  'coco_url': 'http://images.cocodataset.org/val2017/000000160556.jpg', 
#  'height': 428, 
#  'width': 640, 
#  'date_captured': '2013-11-24 05:56:31', 
#  'flickr_url': 'http://farm4.staticflickr.com/3499/3696364224_2996c72d68_z.jpg', 
#  'id': 160556}

I = Image.open('{}/images/{}/{}'.format(dataDir,dataType,img['file_name']))

# print(np.array(I))

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
# print(anns[0])
# {'segmentation': [[569.57, 154.93, 577.96, 149.28, 588.16, 144.36, 593.81, 141.81, 597.27, 141.62, 600.55, 144.54, 602.56, 151.65, 602.74, 157.48, 599.28, 163.13, 593.45, 167.32, 581.05, 175.88, 567.57, 185.36, 573.22, 177.89, 578.5, 172.97, 584.7, 166.41, 585.43, 164.77, 586.16, 162.03, 588.16, 158.39, 589.98, 154.56, 590.45, 150.19, 589.26, 148.45, 584.97, 149.73, 580.24, 153.2, 577.23, 156.38, 576.13, 160.21, 575.77, 161.85, 565.93, 170.05, 564.29, 169.32, 565.02, 167.86, 566.29, 163.85, 570.12, 158.57, 570.12, 155.11]], 
#  'area': 568.7984999999989, 
#  'iscrowd': 0, 
#  'image_id': 160556, 
#  'bbox': [564.29, 141.62, 38.45, 43.74], 
#  'category_id': 77, 'id': 327836}


# filterClasses = ['laptop', 'tv', 'cell phone']
mask = np.zeros((img['height'],img['width']))
for i in range(len(anns)):
    className = getClassName(anns[i]['category_id'], all_cats)
    pixel_value = filterClasses.index(className)+1
    mask = np.maximum(coco.annToMask(anns[i])*pixel_value, mask)
# plt.imshow(mask)
# plt.savefig("tmp.png")