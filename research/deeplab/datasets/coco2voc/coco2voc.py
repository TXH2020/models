from pycocotools.coco import COCO
from coco2voc_aux import *
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import time
import labelme


def coco2voc1(anns_file, root_folder, mode='train', name_bits=7, n=None, compress=True):
    coco_instance = COCO(anns_file)
    coco_imgs = coco_instance.imgs
    if n is None:
        n = len(coco_imgs)
    else:
        assert type(n) == int, "n must be an int"
        n = min(n, len(coco_imgs))

    instance_target_path = os.path.join(root_folder, 'SegmentationObject')
    class_target_path = os.path.join(root_folder, 'SegmentationClass')
    id_target_path = os.path.join(root_folder, 'SegmentationId')
    image_id_list_path = os.path.join(root_folder, 'ImageSets', 'Segmentation', f'{mode}.txt')

    if not os.path.exists(instance_target_path): os.system(f'mkdir {instance_target_path}')
    if not os.path.exists(class_target_path): os.system(f'mkdir {class_target_path}')
    if not os.path.exists(id_target_path): os.system(f'mkdir {id_target_path}')
    image_id_list = open(image_id_list_path, 'a+')

    start = time.time()
    for i, img in enumerate(coco_imgs):
        anns_ids = coco_instance.getAnnIds(img)
        anns = coco_instance.loadAnns(anns_ids)
        if not anns:
            continue

        img = str(img).zfill(name_bits)
        class_seg, instance_seg, id_seg = annsToSeg(anns, coco_instance)

        # Image.fromarray(class_seg).convert("L").save(class_target_path + '/' + str(img) + '.png')
        # Image.fromarray(instance_seg).convert("L").save(instance_target_path + '/' + str(img) + '.png')
        labelme.utils.lblsave(class_target_path + '/' + str(img) + '.png', class_seg)
        labelme.utils.lblsave(instance_target_path + '/' + str(img) + '.png', instance_seg)

        if compress:
            np.savez_compressed(os.path.join(id_target_path, str(img)), id_seg)
        else:
            np.save(os.path.join(id_target_path, str(img) + '.npy'), id_seg)

        image_id_list.write(str(img) + '\n')

        if i % 100 == 0 and i > 0:
            print(str(i) + " annotations processed" +
                  " in " + str(int(time.time() - start)) + " seconds")
        if i >= n:
            break

    image_id_list.close()
    return

def coco2voc2(anns_file, root_folder, mode='train', name_bits=7, n=None, compress=True):
    coco_instance = COCO(anns_file)
    coco_imgs = coco_instance.imgs
    if n is None:
        n = len(coco_imgs)
    else:
        assert type(n) == int, "n must be an int"
        n = min(n, len(coco_imgs))

    instance_target_path = os.path.join(root_folder, 'SegmentationObject1')
    class_target_path = os.path.join(root_folder, 'SegmentationClass1')
    id_target_path = os.path.join(root_folder, 'SegmentationId1')
    image_id_path = os.path.join(root_folder, 'ImageSets1')
    image_id_seg_path=os.path.join(image_id_path, 'Segmentation')
    image_id_list_path = os.path.join(image_id_seg_path,f'{mode}.txt')

    if not os.path.exists(instance_target_path): os.system(f'mkdir {instance_target_path}')
    if not os.path.exists(class_target_path): os.system(f'mkdir {class_target_path}')
    if not os.path.exists(id_target_path): os.system(f'mkdir {id_target_path}')
    if not os.path.exists(image_id_path): os.system(f'mkdir {image_id_path}')
    if not os.path.exists(image_id_seg_path): os.system(f'mkdir {image_id_seg_path}')
    image_id_list = open(image_id_list_path, 'a+')

    start = time.time()
    for i, img in enumerate(coco_imgs):
        anns_ids = coco_instance.getAnnIds(img)
        anns = coco_instance.loadAnns(anns_ids)
        if not anns:
            continue

        img = str(img).zfill(name_bits)
        class_seg, instance_seg, id_seg = annsToSeg(anns, coco_instance)

        # Image.fromarray(class_seg).convert("L").save(class_target_path + '/' + str(img) + '.png')
        # Image.fromarray(instance_seg).convert("L").save(instance_target_path + '/' + str(img) + '.png')
        labelme.utils.lblsave(class_target_path + '/' + str(img) + '.png', class_seg)
        labelme.utils.lblsave(instance_target_path + '/' + str(img) + '.png', instance_seg)

        if compress:
            np.savez_compressed(os.path.join(id_target_path, str(img)), id_seg)
        else:
            np.save(os.path.join(id_target_path, str(img) + '.npy'), id_seg)

        image_id_list.write(str(img) + '\n')

        if i % 100 == 0 and i > 0:
            print(str(i) + " annotations processed" +
                  " in " + str(int(time.time() - start)) + " seconds")
        if i >= n:
            break

    image_id_list.close()
    return
