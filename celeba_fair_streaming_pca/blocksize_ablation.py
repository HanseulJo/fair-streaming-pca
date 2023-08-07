import os
import torch
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from fair_streaming_pca_pytorch import FairStreamingPCA
import warnings
warnings.filterwarnings('ignore')


def main(attr_name, dataset_train, dataset_val):
    n_iter_list = [0,5,20,50,100]
    batch_size_train = 1000
    batch_size_val = 1000
    device = 'mps'
    num_workers = 0
    k = 1000
    verbose = False

    ATTR = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
    ATTR = {attr: attr_idx for attr_idx, attr in enumerate(ATTR.split(' '))}
    
    loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, drop_last=False)

    # a = attr
    if not os.path.isdir(f'figures/{attr_name}_{ATTR[attr_name]}/'):
        os.makedirs(f'figures/{attr_name}_{ATTR[attr_name]}/')
    attr_true_list_val = []
    attr_false_list_val = []
    cnt = [20, 20]
    for i, (im, lab) in enumerate(dataset_val):
        l = lab[ATTR[attr_name]].item()
        if l==1 and cnt[1] > 0:
            attr_true_list_val.append(i)
            cnt[1] -= 1
        elif l==0 and cnt[0] > 0:
            attr_false_list_val.append(i)
            cnt[0] -= 1
        if sum(cnt) == 0 : break
    print('Target Attribute: ', attr_name)
    print('True: ', attr_true_list_val)
    print('False: ', attr_false_list_val)

    V_list = []
    N_list = []
    # exp_var_list = []
    # exp_var_0_list = []
    # exp_var_1_list = []
    print("Start fitting")
    for n_iter in n_iter_list:
        block_size = None if n_iter==0 else 160000 // n_iter
        pca = FairStreamingPCA(attr_name, device=device)
        pca.fit(
            dataset=dataset_train,
            target_unfair_dim=5,
            target_pca_dim=k,
            n_iter_unfair=n_iter,
            n_iter_pca=5,
            block_size_unfair=block_size,
            block_size_pca=32000,
            batch_size=batch_size_train,
            constraint='vanilla' if n_iter==0 else 'all',
            verbose=verbose,
            seed=0
        )
        V_list.append(pca.V.cpu())
        N_list.append(pca.N.cpu())
        # pca.transform(loader=loader_val,)
        # exp_var_list.append(pca.explained_variance_ratio)
        # exp_var_0_list.append(pca.explained_variance_ratio_group[0])
        # exp_var_1_list.append(pca.explained_variance_ratio_group[1])


    # fig, ax = plt.subplots()
    # ax.plot(list(map(str, rank_list)), exp_var_list, label='ExpVar', marker='*', markersize=10)
    # ax.plot(list(map(str, rank_list)), exp_var_0_list, label='ExpVar0', marker='x')
    # ax.plot(list(map(str, rank_list)), exp_var_1_list, label='ExpVar1', marker='o')
    # ax.legend()
    
    # fig.savefig(f'figures/{attr_name}_{ATTR[attr_name]}/RankAblation_expvar.pdf')
    # plt.close(fig)

    denorm = transforms.Normalize((-1,),(2,))
    def im_show(img, ax=None):
        img_denorm = denorm(img)
        img_t = torch.permute(img_denorm, (1,2,0))
        img_t.clamp_(0, 1)
        if ax is not None:
            ax.imshow(img_t)
        else:
            plt.imshow(img_t)
    
    # print(attr_false_list_val + attr_true_list_val)
    for index in attr_false_list_val+attr_true_list_val:
        fig, axes = plt.subplots(2, len(n_iter_list), figsize=(2*len(n_iter_list), 5))
        axes[0][0].set_ylabel('Span')
        axes[1][0].set_ylabel('Orthogonal')

        img, label = dataset_val[index]
        im_show(img, axes[0][0])
        for i, n_iter in enumerate(n_iter_list):
            pca.V = V_list[i].to(pca.device)
            projected = pca.transform(img).cpu()
            im_show(projected[0], axes[0][i])
            if n_iter == 0:
                axes[0][i].title.set_text(f'Vanilla')
                im_show(img, axes[1][i])
                axes[1][i].set_xlabel('(original)')
            else:
                pca.V = N_list[i].to(pca.device)
                orthogonal = pca.transform(img).cpu()
                im_show(orthogonal[0], axes[1][i])
                axes[0][i].title.set_text(f'b={160000//n_iter}')
            axes[0][i].get_xaxis().set_ticks([])
            axes[0][i].get_yaxis().set_ticks([])
            axes[1][i].get_xaxis().set_ticks([])
            axes[1][i].get_yaxis().set_ticks([])

        fig.tight_layout()
        fig.savefig(f'figures/{attr_name}_{ATTR[attr_name]}/BlockSizeAblation_all5_{bool(label[pca.a])}_{index}.pdf')
        plt.close(fig)
    del pca

if __name__ == '__main__':
    root = 'datasets/'
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    print("\nloading dataset...")
    # order = list(range(1600))
    # dataset_train = Subset(CelebA(root=root, split='train', transform=transform, target_type='attr', download=False), order)
    dataset_train = CelebA(root=root, split='train', transform=transform, target_type='attr', download=False)
    dataset_val = CelebA(root=root, split='valid', transform=transform, target_type='attr', download=False)

    #attr_list = "Attractive Wearing_Lipstick Wearing_Necklace Wearing_Necktie "
    #attr_list = "Eyeglasses Goatee Bangs Narrow_Eyes"
    attr_list = "Smiling Mouth_Slightly_Open Male Mustache No_Beard Wearing_Hat Young Bald Black_Hair Blond_Hair Brown_Hair "
    attr_list = attr_list.split(' ')
    dataset_list_train = [dataset_train for _ in range(len(attr_list))]
    dataset_list_val = [dataset_val for _ in range(len(attr_list))]

    mp.set_start_method("spawn")
    num_processes = 1
    processes = []
    with mp.Pool(processes=num_processes) as p:
        ret = p.starmap_async(main, zip(attr_list, dataset_list_train, dataset_list_val))
        ret.get()
        # We first train the model across `num_processes` processes
        
