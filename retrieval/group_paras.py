import numpy as np
import faiss


def write_file(file_name, samples):
    with open(file_name, 'w') as f_out:
        for _ in samples:
            f_out.write(_)


def group_paras(I, ncentroids, split_path='/home/xwhan/retrieval_data/splits_512/'):
    samples = [[] for _ in range(ncentroids)]
    with open('/data/hongwang/nq_rewrite/db/re_data/retrieve_train.txt') as f_in:
        for i, line in enumerate(f_in):
            samples[I[i][0]].append(line)
    for i, group in enumerate(samples):
        write_file( split_path + 'split_'+str(i)+'.txt', group)

def clusering(data, niter=1000, verbose=True, ncentroids=1024, max_points_per_centroid=10000000, gpu_id=0, spherical=False):
    # use one gpu
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = gpu_id
    
    d = data.shape[1]
    if spherical:   
        index = faiss.GpuIndexFlatIP(res, d, cfg)
    else:
        index = faiss.GpuIndexFlatL2(res, d, cfg)

    clus = faiss.Clustering(d, ncentroids)
    clus.verbose = True
    clus.niter = niter
    clus.max_points_per_centroid = max_points_per_centroid

    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)
    centroids = centroids.reshape(ncentroids, d)

    index.reset()
    index.add(centroids)
    D, I = index.search(data, 1)

    return D, I

if __name__ == "__main__":
    train_para_embed_path = "/mnt/edward/home/xwhan/retrieval_data/embed/train_para_embed_3_28_c10000.npy"
    split_save_path = "/home/xwhan/retrieval_data/final_splits_spherical/"

    x = np.load(train_para_embed_path)
    x = np.float32(x)

    ncentroids = 10000
    niter = 250
    max_points_per_centroid = 1000
    spherical = True

    D, I = clusering(x, niter=niter, ncentroids=ncentroids, max_points_per_centroid=max_points_per_centroid, spherical=spherical)

    group_paras(I, ncentroids, split_path=split_save_path)
