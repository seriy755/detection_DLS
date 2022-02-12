from glob import glob
import os
import os.path as osp

home = osp.expanduser("~")
get_dir = '\project_SSD\data'
data_root = home+get_dir

roots = ['train', 'test']
anno_roots = ['train_anno', 'test_anno']
for i, root in enumerate(roots):
    path = osp.join(data_root, root)
    anno_files = glob(path+'\*.xml')
    for file in anno_files:
        path = osp.join(data_root, anno_roots[i])
        new_name = file.split('\\')[-1]
        os.replace(file, osp.join(path, new_name))