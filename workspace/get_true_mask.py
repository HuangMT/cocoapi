import json
from pycocotools.coco import COCO
import numpy as np
import cv2
root_path = '/home/mingtao.huang/'
dataDir=root_path + 'dataset/coco/'
dataType='train2017'
img_path=dataDir+dataType+'/'
annFile='{}annotations/instances_{}.json'.format(dataDir,dataType)
coco = COCO(annFile)
with open('choice.json','r') as jf:
    j = jf.readlines()[0]
    dic = json.loads(j)
    dic['foreids'] = coco.getCatIds(catNms=dic['foreground'])
    dic['backids'] = coco.getCatIds(catNms=dic['background'])
    dic['condids'] = coco.getCatIds(catNms=dic['conditional'])
    print(dic)
def getForegroundMask(img_id):
    img_msg = coco.loadImgs(ids=img_id)[0]
    C = np.zeros((img_msg['height'],img_msg['width']))
    area = 0
    for ann in coco.loadAnns(ids=coco.getAnnIds(imgIds=img_id)):
        #print(ann)
        if ann['category_id'] in dic['foreids']:
            C += coco.annToMask(ann)>0
            area += ann['area']
    for ann in coco.loadAnns(ids=coco.getAnnIds(imgIds=img_id)):
        if ann['category_id'] in dic['condids']:
            mask = coco.annToMask(ann).astype(np.uint8)
            if np.sum(C[cv2.dilate(mask,np.ones((7,7),np.uint8))>0]) >0:
                C += mask
                area += ann['area']
    return area,(C>0).astype(np.uint8)
cnt = 0
with open('{}log/available_imgid_{}.txt'.format(dataDir,dataType),'w') as f:
    for img_id in coco.getImgIds():
        cnt += 1
        if cnt % 100==0:
            print('transformed %6d pictures.'%cnt)
        area,mask = getForegroundMask(img_id)
        if area>1000:
            cv2.imwrite(dataDir+'coco_mask_'+dataType+'/%012d.png'%coco.loadImgs(ids=img_id)[0]['id'],mask)
            f.writelines(str(img_id)+'\n')
print('%6d in total'%cnt)