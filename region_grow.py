import math
import numpy as np
from sklearn.neighbors import KDTree

unique_rows=np.loadtxt("model_pointcloud.txt")
tree    = KDTree(unique_rows, leaf_size=2) 
dist,nn_glob = tree.query(unique_rows[:len(unique_rows)], k=30) 

def normalsestimation(pointcloud,nn_glob,VP=[0,0,0]):
    ViewPoint = np.array(VP)
    normals = np.empty((np.shape(pointcloud)))
    curv    = np.empty((len(pointcloud),1))
    for index in range(len(pointcloud)):
        nn_loc = pointcloud[nn_glob[index]]
        COV = np.cov(nn_loc,rowvar=False)
        eigval, eigvec = np.linalg.eig(COV)
        idx = np.argsort(eigval)
        nor = eigvec[:,idx][:,0]
        if nor.dot((ViewPoint-pointcloud[index,:])) > 0:
            normals[index] = nor
        else:
            normals[index] = - nor
        curv[index] = eigval[idx][0] / np.sum(eigval)
    return normals,curv


def regiongrowing(pointcloud,nn_glob,theta_th = 'auto', cur_th = 'auto'):
    normals,curvature = normalsestimation(pointcloud,nn_glob=nn_glob)
    order             = curvature[:,0].argsort().tolist()
    region            = []
    if theta_th == 'auto':
        theta_th          = 15.0 / 180.0 * math.pi 
    if cur_th == 'auto':
        cur_th            = np.percentile(curvature,98)
    while len(order) > 0:
        region_cur = []
        seed_cur   = []
        poi_min    = order[0] 
        region_cur.append(poi_min)
        seed_cur.append(poi_min)
        order.remove(poi_min)
        for i in range(len(seed_cur)):
            nn_loc  = nn_glob[seed_cur[i]]
            for j in range(len(nn_loc)):
                nn_cur = nn_loc[j]
                if all([nn_cur in order , np.arccos(np.abs(np.dot(normals[seed_cur[i]],normals[nn_cur])))<theta_th]):
                    region_cur.append(nn_cur)
                    order.remove(nn_cur)
                    if curvature[nn_cur] < cur_th:
                        seed_cur.append(nn_cur)
        region.append(region_cur)
    return region

def regiongrowing1(pointcloud,nn_glob,theta_th = 'auto', cur_th = 'auto'):
    normals,curvature = normalsestimation(pointcloud,nn_glob=nn_glob)
    order             = curvature[:,0].argsort().tolist()
    region            = []
    if theta_th == 'auto':
        theta_th          = 15.0 / 180.0 * math.pi 
    if cur_th == 'auto':
        cur_th            = np.percentile(curvature,98)
    while len(order) > 0:
        region_cur = []
        seed_cur   = []
        poi_min    = order[0] 
        region_cur.append(poi_min)
        seedval = 0

        seed_cur.append(poi_min)
        order.remove(poi_min)
        while seedval < len(seed_cur):
            nn_loc  = nn_glob[seed_cur[seedval]]
            for j in range(len(nn_loc)):
                nn_cur = nn_loc[j]
                if all([nn_cur in order , np.arccos(np.abs(np.dot(normals[seed_cur[seedval]],normals[nn_cur])))<theta_th]):
                    region_cur.append(nn_cur)
                    order.remove(nn_cur)
                    if curvature[nn_cur] < cur_th:
                        seed_cur.append(nn_cur)
            seedval+=1
        region.append(region_cur)
    return region

region= regiongrowing(unique_rows,nn_glob)
