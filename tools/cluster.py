from sklearn.metrics import pair_confusion_matrix
import torch
import torch.nn.functional as F
from sklearn import manifold
import tqdm
import numpy as np
from munkres import Munkres, print_matrix
from sklearn.cluster import (KMeans, AffinityPropagation, MeanShift, estimate_bandwidth, DBSCAN,
                             SpectralClustering, AgglomerativeClustering)
from sklearn.mixture import GaussianMixture

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
matplotlib.rc('text', usetex = True)

from matplotlib import rcParams

config = {
            "font.family": 'serif',
            "font.size": 15,
            "mathtext.fontset": 'stix',
            # "font.serif": ['SimSun'],
            "font.serif": ['Times New Roman'],
            'axes.unicode_minus': False # 处理负号，即-号
         }
rcParams.update(config)

pth_dir = 'checkpoints/voc/exp_gdl1_voc_CosSimNegHead/defrcn_det_r101_base1/'
# pth_dir = 'work_dirs/fgdet/ucow_oln_box/kuc_em_csh/DE/wo/'
# pth_dir = 'work_dirs/fgdet/ucow_oln_box/kuc_em_csh/DE/csh/'
# pth_dir = 'work_dirs/fgdet/ucow_oln_box/kuc_em_csh/DE/em_csh/'

classes = ["bird", "bus", "cow", "motorbike", "sofa"]

def tsne(vectors, weights, center_features=None, plot=False):
    # all_class_features = torch.cat(vectors, dim=0)
    # all_class_features = F.normalize(all_class_features)
    vectors = vectors.cpu().numpy()
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()
    if center_features is not None:
        center_features = center_features.cpu().numpy()

    tsne = manifold.TSNE(n_components=2, init='pca', metric='cosine', perplexity=50, n_iter=5000, square_distances=True)

    if center_features is not None:
        num_cluster = center_features.shape[0]
        cat_features = np.concatenate((vectors, center_features), axis=0)
        X_tsne = tsne.fit_transform(cat_features)
    else:
        X_tsne = tsne.fit_transform(vectors)
    # torch.save(X_tsne, X_tsne_path)
    print('tsne finish')

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    if center_features is not None:
        center_norm = X_norm[-num_cluster:, :]
        X_norm = X_norm[:-num_cluster, :]
    else:
        center_norm = None
    # plt.figure(100, figsize=(5.5, 4))
    # plt.plot(X_norm[:, 0], X_norm[:, 1], marker='.', color='red', linestyle='', markersize=10)
    if plot:
        tsne_show(X_norm, weights, 0., 1., center_norm)
    return X_norm, center_norm

def tsne_show(X_norm, weights=None, vmin=0., vmax=1.0, center_norm=None):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    if weights is not None:
        if isinstance(weights, torch.Tensor):
            weights = weights.cpu().numpy()
        plt.scatter(X_norm[:, 0], X_norm[:, 1], c=weights, marker='.', cmap='rainbow', norm=norm, s=70)
    else:
        plt.scatter(X_norm[:, 0], X_norm[:, 1], marker='.', cmap='rainbow', norm=norm, s=70)
    # plt.colorbar(fraction=0.05, pad=0.01)
    # plt.legend(vectors, loc='lower left', ncol=2, handlelength=1.0, handletextpad=0.2, columnspacing=0.5)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xticks([])
    plt.yticks([])
    if center_norm is not None:
        plt.scatter(center_norm[:, 0], center_norm[:, 1], color='black', marker='s', s=120)
    # plt.show()


def cluster(instance_features, cls_score, num_neighbor=20, center_thresh=0.65, mean_thresh=None, show=False, labels=None):
    num_instance = instance_features.size(0)

    all_class_features_ex1 = instance_features[None, :, :].expand(num_instance, -1, -1)
    all_class_features_ex2 = instance_features[:, None, :].expand_as(all_class_features_ex1)

    cos_sim_mat = F.cosine_similarity(all_class_features_ex1, all_class_features_ex2, dim=2)
    cos_sim_score = cos_sim_mat.topk(num_neighbor, dim=1)[0].relu().mean(1)

    # get class centers
    class_center_inds = []
    max_val, max_ind = cos_sim_score.max(0)
    cos_sim_score_list = []
    while max_val > center_thresh:
        class_center_inds.append(max_ind.item())
        cos_sim_score_list.append(cos_sim_score)
        cos_sim_score = cos_sim_score * torch.sub(1, cos_sim_mat[:, max_ind].relu() ** 2)
        max_val, max_ind = cos_sim_score.max(0)


    num_cluster = len(class_center_inds)
    print(num_cluster)
    center_features = instance_features[class_center_inds, :]
    cls_id = cos_sim_mat[:, class_center_inds].max(1)[1]
    last_cls_id = cls_id

    if show:
        X_norm, center_norm = tsne(instance_features, cos_sim_score, center_features)
        for i, css in enumerate(cos_sim_score_list):
            tsne_show(X_norm, weights=css, vmin=0., vmax=1.0)
            plt.scatter(center_norm[:i+1, 0], center_norm[:i+1, 1], color='black', marker='s', s=100)
            plt.subplots_adjust(left=0.0, right=0.94, top=0.97, bottom=0.03)
            plt.show()

        # tsne_show(X_norm, weights=cos_sim_score_list[0], vmin=0., vmax=1.0)
        # plt.scatter(center_norm[:5, 0], center_norm[:5, 1], color='black', marker='s', s=100)
        # plt.subplots_adjust(left=0.0, right=0.94, top=0.97, bottom=0.03)
        # plt.show()
    # tsne(instance_features, cls_id * (1. / num_cluster), center_features=center_features)

    max_step = 100
    for i in tqdm.tqdm(range(max_step)):
        # if i == 6:
        #     X_norm, center_norm = tsne(instance_features, None, center_features)
        #     cls_id_map = best_map(labels.cpu().numpy(), cls_id.cpu().numpy())
        #     tsne_show(X_norm, cls_id_map * (1. / cls_id_map.max()))
        #     plt.scatter(center_norm[:5, 0], center_norm[:5, 1], color='black', marker='s', s=100)
        #     plt.title('$iter={}$'.format(i+1), fontdict={'size':30})
        #     plt.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.0)
        #     plt.show()

        center_features = []
        for cluster_id in range(num_cluster):
            center_features.append(instance_features[cls_id==cluster_id].mean(0, keepdim=True))
        center_features = torch.cat(center_features, dim=0)
        cos_sim = F.cosine_similarity(
            instance_features[:, None, :].expand(-1, num_cluster, -1),
            center_features[None, :, :].expand(num_instance, -1, -1),
            dim=-1
        )
        cls_id = cos_sim.max(1)[1]
        if cls_id.eq(last_cls_id).all():
            break
        last_cls_id = cls_id


    return cls_id, num_cluster, center_features

def select_instance(instance_features, cls_logits, n=50):
    all_class_features = []
    all_class_nums = []
    all_class_names = []
    all_scores = []
    num_features = dict()
    for c in classes:
        num_features[c] = 0

    labels = []
    for k, v in instance_features.items():
        if k == 'unknown':
            continue
        if k not in classes:
            continue
        # if num_features[k] > 2:
        #     continue
        # num_features[k] += 1

        all_class_names.append(k)
        all_class_nums.append(len(v))
        # all_class_features.append(torch.cat(v, dim=0))
        v = torch.cat(v, dim=0)
        num = min(v.size(0), n)
        all_class_features.append(v[:num])
        # all_scores.append(cls_logits[k].softmax(1)[:num, :-1].max(1)[0])
        labels.append(torch.ones(num, dtype=torch.int) * classes.index(k))

    labels = torch.cat(labels)
    all_class_features = torch.cat(all_class_features, dim=0)
    # all_scores = torch.cat(all_scores, dim=0)
    return all_class_features, all_scores, labels

def accuracy(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]

def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    ri = (tp + tn) / (tp + tn + fp + fn)
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = 2*p*r/(p+r)
    return ri, f_beta

def best_map(L1, L2):
    #L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)       # 去除重复的元素，由小大大排列
    nClass1 = len(Label1)        # 标签的大小
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def estimate_cluster(y, y_pred):
    purity = accuracy(y, y_pred)
    ri, f_beta = get_rand_index_and_f_measure(y, y_pred, beta=1.)
    print(f"聚类纯度：{purity}\n兰德系数：{ri}\nF1值：{f_beta}")

# cluster_setting = (100, 20, 0.58)
cluster_setting = (200, 20, 0.7)

def main():
    pth_file = pth_dir + 'voc_test_instance_features.pth'
    outputs = torch.load(pth_file, map_location='cpu')
    cls_logits = None
    x_after_fc = outputs
    # tsne_vectors(x_after_fc, read_from_file=False)
    instance_features, cls_scores, labels = select_instance(x_after_fc, cls_logits, n=cluster_setting[0])
    instance_features = F.normalize(instance_features, p=2, dim=-1)

    X_norm, _  = tsne(instance_features, None, plot=False)

    print('CSC')
    cluster_res, num_cluster, center_features = cluster(F.normalize(instance_features, p=2, dim=-1), cls_scores, num_neighbor=cluster_setting[1], center_thresh=cluster_setting[2], show=False, labels=labels)
    cluster_res = best_map(labels.cpu().numpy(), cluster_res.cpu().numpy())
    estimate_cluster(labels, cluster_res)
    # tsne(instance_features, cluster_res * (1. / cluster_res.max()), plot=True)
    tsne_show(X_norm, cluster_res * (1. / cluster_res.max()))
    # plt.title('CSC', fontdict={'size':40}, fontfamily='Times New Roman')
    plt.show()

    print('K means')
    k_means_pre = KMeans(5, max_iter=3000).fit_predict(F.normalize(instance_features, p=2, dim=-1).cpu().numpy())
    k_means_pre = best_map(labels.cpu().numpy(), k_means_pre)
    estimate_cluster(labels, k_means_pre)
    # tsne(instance_features, k_means_pre * (1. / k_means_pre.max()), plot=True)
    tsne_show(X_norm, k_means_pre * (1. / k_means_pre.max()))
    # plt.title('K-means', fontdict={'size':40}, fontfamily='Times New Roman')
    plt.show()

    print('affinity_propagation')
    affinity_propagation_pre = AffinityPropagation(damping=0.7, random_state=0,max_iter=300).fit_predict(F.normalize(instance_features, p=2, dim=-1).cpu().numpy())
    print(affinity_propagation_pre.min(), affinity_propagation_pre.max())
    # tsne(instance_features, affinity_propagation_pre * (1. / (affinity_propagation_pre.max())), plot=True)
    tsne_show(X_norm, affinity_propagation_pre * (1. / affinity_propagation_pre.max()))
    # plt.title('近邻传播', fontdict={'size':40})
    plt.show()
    #q
    # print('mean_shift')
    # # bandwidth = estimate_bandwidth(instance_features.cpu().numpy(), quantile=0.5, n_samples=1000)
    # mean_shift_pre = MeanShift(bandwidth=0.5, max_iter=300).fit_predict(F.normalize(instance_features, p=2, dim=-1).cpu().numpy())
    # # mean_shift_pre = MeanShift(max_iter=300).fit_predict(instance_features.cpu().numpy())
    # print(mean_shift_pre.min(), mean_shift_pre.max())
    # # mean_shift_pre = best_map(labels.cpu().numpy(), mean_shift_pre)
    # # estimate_cluster(labels, mean_shift_pre)
    # tsne(instance_features, mean_shift_pre * (1. / (mean_shift_pre.max())), plot=True)
    # plt.title('均值漂移', fontdict={'size':40})
    # plt.show()

    print('DBSCAN')
    DBSCAN_pre = DBSCAN(eps=0.7, min_samples=30).fit_predict(F.normalize(instance_features, p=2, dim=-1).cpu().numpy())
    DBSCAN_centers = []
    for i in range(5):
        DBSCAN_centers.append(instance_features[DBSCAN_pre==i].mean(0, keepdim=True))
    DBSCAN_centers = torch.cat(DBSCAN_centers, dim=0)
    _, DBSCAN_pre = F.cosine_similarity(instance_features[:, None, :].expand(-1, 5, -1), DBSCAN_centers[None, :, :].expand(instance_features.size(0), -1, -1), dim=-1).max(-1)
    DBSCAN_pre = DBSCAN_pre.cpu().numpy()
    DBSCAN_pre = best_map(labels.cpu().numpy(), DBSCAN_pre)
    estimate_cluster(labels, DBSCAN_pre)
    # tsne(instance_features, DBSCAN_pre * (1. / (DBSCAN_pre.max())), plot=True)
    tsne_show(X_norm, DBSCAN_pre * (1. / DBSCAN_pre.max()))
    # plt.title('DBSCAN', fontdict={'size':40}, fontfamily='Times New Roman')
    plt.show()
    # DBSCAN_pre = DBSCAN(eps=20., min_samples=2).fit_predict(instance_features.cpu().numpy())
    # print(DBSCAN_pre.shape, DBSCAN_pre.min(), DBSCAN_pre.max())
    # tsne(instance_features, (DBSCAN_pre + 1) * (1. / (DBSCAN_pre.max() + 1)), plot=True)
    # plt.show()

    print('SpectralClustering')
    SpectralClustering_pre = SpectralClustering(5).fit_predict(F.normalize(instance_features, p=2, dim=-1).cpu().numpy())
    SpectralClustering_pre = best_map(labels.cpu().numpy(), SpectralClustering_pre)
    estimate_cluster(labels, SpectralClustering_pre)
    # tsne(instance_features, SpectralClustering_pre * (1. / (SpectralClustering_pre.max())), plot=True)
    tsne_show(X_norm, SpectralClustering_pre * (1. / SpectralClustering_pre.max()))
    # plt.title('谱聚类', fontdict={'size':40})
    plt.show()

    print('AgglomerativeClustering')
    AgglomerativeClustering_pre = AgglomerativeClustering(5, ).fit_predict(F.normalize(instance_features, p=2, dim=-1).cpu().numpy())
    AgglomerativeClustering_pre = best_map(labels.cpu().numpy(), AgglomerativeClustering_pre)
    estimate_cluster(labels, AgglomerativeClustering_pre)
    tsne(instance_features, AgglomerativeClustering_pre * (1. / (AgglomerativeClustering_pre.max())), plot=True)
    # plt.title('层次聚类', fontdict={'size':40})
    plt.show()
    # tsne(instance_features, AgglomerativeClustering_pre * (1. / num_cluster), plot=True)

    # affinity_propagation_pre = best_map(labels.cpu().numpy(), affinity_propagation_pre)
    # estimate_cluster(labels, affinity_propagation_pre)

    # tsne(instance_features, cluster_res * (1. / num_cluster), center_features=center_features, plot=True)
    # plt.show()
    # tsne(instance_features, labels * (1. / num_cluster), plot=True)
    # plt.show()

    # tsne(instance_features, labels * (1. / 4), plot=True)
    tsne_show(X_norm, labels * (1. / 4))
    # plt.title('真实标签', fontdict={'size':40})
    plt.show()
    # tsne(instance_features, k_means_pre * (1. / num_cluster))

if __name__ == '__main__':
    main()