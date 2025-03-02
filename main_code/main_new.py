import os
import argparse
import pickle

import h5py
from torch.utils.data import DataLoader
import wandb
import torch.nn as nn
import random

from read_data import *
from utils import patient_kfold 
from vit_new import train, ViT, evaluate
from summarymixer import ViS, ViSPriorKnowledgeBalanced, ViSPriorKnowledgeUnbalanced

import json

def custom_collate_fn(batch):
    """Remove bad entries from the dataloader
    Args:
        batch (torch.Tensor): batch of tensors from the dataaset
    Returns:
        collate: Default collage for the dataloader
    """
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def filter_no_features(df, feature_path):
    print("[LOG] Extracting features and filtering WSI's without features")
    projects = np.unique(df.tcga_project)
    all_wsis_with_features = []
    remove = []
    for proj in projects:
        wsis_with_features = os.listdir(feature_path)
        #wsis_with_features = os.listdir(feature_path + proj)

        # filter the ones without cluster_features
        for wsi in wsis_with_features:
            try:
                #with h5py.File(feature_path +proj+ '/'+wsi+'/'+wsi+'.h5', "r") as f:
                with h5py.File(feature_path + '/' + wsi, "r") as f:
                    cols = list(f.keys())
                    if 'cluster_features' not in cols:
                        print(f"[LOG] removed {wsi} due to no 'cluster_features'")
                        remove.append(wsi)
            except Exception as e:
                print(f"[LOG] removed {wsi} due to {e}")
                remove.append(wsi)

        #print(len(remove))
        all_wsis_with_features += [name.split('.h5')[0] for name in wsis_with_features if name not in remove]
        #all_wsis_with_features += wsis_with_features
        #print(all_wsis_with_features)

    remove += df[~df['wsi_file_name'].isin(all_wsis_with_features)].wsi_file_name.values.tolist()
    #print(remove)
    #print(len(remove))
    print(f'[LOG] Original shape: {df.shape}')
    df = df[~df['wsi_file_name'].isin(remove)].reset_index(drop=True)
    print(f'[LOG] New shape: {df.shape}')
    return df

def filter_genes(data, selected_genes):
    selected_genes = [f'rna_{g}' for g in selected_genes]
    retain_cols = []
    for col in data.columns:
        if 'rna_' not in col:
            retain_cols.append(col)
            continue
        if 'rna_' in col and col in selected_genes:
            retain_cols.append(col)
    f_data = data.loc[:, retain_cols]
    return f_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')

    # general args
    parser.add_argument('--src_path', type=str, default='.', help='project path')
    parser.add_argument('--ref_file', type=str, default=None, help='path to reference file')
    parser.add_argument('--tcga_projects', help="the tcga_projects we want to use, separated by comma", default=None, type=str)
    parser.add_argument('--feature_path', type=str, default="/oak/stanford/groups/ogevaert/data/Gen-Pred/features_resnet/", help='path to resnet and clustered features')
    parser.add_argument('--feature_dim', type=int, default=2048, help='kmean feature dimension')
    parser.add_argument('--save_dir', type=str, default='vit_exp', help='parent destination folder')
    parser.add_argument('--cohort', type=str, default="TCGA", help='cohort name for creating the saving folder of the results')
    parser.add_argument('--exp_name', type=str, default="exp", help='Experiment name for creating the saving folder of the results')
    parser.add_argument('--log', type=int, default=1, help='Whether to log the loss during training')

    # model args
    parser.add_argument('--model_type', type=str, default='vis', help='"vit" or "vis", depending on desired attention block')
    parser.add_argument('--depth', type=int, default=6, help='transformer depth')
    parser.add_argument('--num-heads', type=int, default=16, help='number of attention heads')
    parser.add_argument('--seed', type=int, default=99, help='Seed for random generation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint from trained model. If tcga_pretrain is true, then the model paths in different folds should be of the format args.checkpoint + "{fold}.pt" ')
    parser.add_argument('--train', help="if you want to train the model", action="store_true")
    parser.add_argument('--filter_genes', type=str, default=None, help='path to a npy file containing a list of genes of interest for training')
    parser.add_argument('--change_num_genes', help="whether finetuning from a model trained on different number of genes", action="store_true")
    parser.add_argument('--num_genes', type=int, default=25761, help='number of genes on which pretrained model was trained')
    parser.add_argument('--k', type=int, default=3, help='Number of splits')
    parser.add_argument('--load_splits', type=int, default=0, help='whether to load splits from checkpoint path')
    parser.add_argument('--stratify', help="stratify k-fold for cancer type, only relevant if training on multiple cancer types", action="store_true")
    parser.add_argument('--save_on', type=str, default='loss', help='which criterium to save model on, "loss" or "loss+corr"')
    parser.add_argument('--stop_on', type=str, default='loss', help='which criterium to do early stopping on, "loss" or "loss+corr"')

    # Prior Knowledge arguments
    parser.add_argument('--model_version', type=str, default='Unbalanced', help='which architecture to use: Balanced or Unbalanced or Regularized')
    parser.add_argument('--priorknowledge_type', type=str, default="external", help="external or internal")
    parser.add_argument('--gene_embeddings_path', type=str, default='gene_embeddings/', help='path to the folder that contains gene embeddings')
    parser.add_argument('--gene_embeddings', type=str, default='embeddings_BRCA_CE_normalized.npy', help='path to the file that contains the gene embeddings')
    parser.add_argument('--lam', type=float, default=0, help='value of the lambda hyperparameter')
    parser.add_argument('--r_threshold', type=float, default = 1, help="threshold of correlation for co-expression when using internal prior knowledge. Options are: 0.85, 0.75, 0.65")

    # testing
    parser.add_argument('--testing', type=int, default=0, help='decrease the amount of slides to run a quick test')

    ############################################## variables ##############################################
    args = parser.parse_args()

    ############################################## seeds ##############################################
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2" # needed for torch.use_determ (below) --> still throws the same error
    torch.backends.cudnn.benchmark = False # possibly reduced performance but better reproducibility
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)

    # reproducibility train dataloader
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(0)

    print(f"[LOG] using seed: {g}")

    ############################################## logging ##############################################
    save_dir = os.path.join(args.src_path, args.save_dir, args.cohort, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run = None
    if args.log:
        print("[LOG] trying to login wandb")
        wandb.login(key="3e7959a551c0e9b3c350df33b741771da9df8ea7")
        run = wandb.init(project="thesis", entity='maxhallemeesch', config=args, name=args.exp_name) 
    
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(f"[LOG] Using device {device}")

    parameters = {
        "model_type":args.model_type,
        "model_version":args.model_version,
        "priorknowledge":args.priorknowledge_type,
        "embeddings":"NMF",
        "lambda":args.lam
    }

    with open(os.path.join(save_dir, 'parameters.json'), 'w') as f:
        json.dump(parameters, f)

    print(f"[LOG] Logging parameters file in json format")

    ############################################## data prep ##############################################
    path_csv = args.ref_file
    df = pd.read_csv(path_csv)

    # testing purposes
    if args.testing:
        df = df.iloc[:20]
        num_epochs = 5
        print("[LOG] reading 20 WSI's")

    # filter tcga projects
    if ('tcga_project' in df.columns) and (args.tcga_projects != None):
        projects = args.tcga_projects.split(',')
        df = df[df['tcga_project'].isin(projects)].reset_index(drop=True)

    print("[LOG] filtering slides whose cluster features do not exist")
    df = filter_no_features(df, args.feature_path)

    # filter genes of interest
    selected_genes = None
    if args.filter_genes is not None:
        selected_genes = np.load(args.filter_genes, allow_pickle=True)
        selected_genes = list(set(selected_genes))
        df = filter_genes(df, selected_genes)
        print(f"Training only for selected genes: n = {len(selected_genes)}")

    ############################################## train, val, test split ##############################################
    if not args.load_splits:
        train_idxs, val_idxs, test_idxs = patient_kfold(df, n_splits=args.k)
    else:
        train_idxs, val_idxs, test_idxs = [], [], []
        for i in range(args.k):
            train_idxs.append(np.load(os.path.join(args.src_path, args.save_dir, args.cohort, args.checkpoint, 'train_'+str(i)+'.npy')), allow_pickle=True)
            val_idxs.append(np.load(os.path.join(args.src_path, args.save_dir, args.cohort, args.checkpoint, 'val_'+str(i)+'.npy')), allow_pickle=True)
            test_idxs.append(np.load(os.path.join(args.src_path, args.save_dir, args.cohort, args.checkpoint, 'test_'+str(i)+'.npy')), allow_pickle=True)
    
    # extract gene embeddings once if type of prior knowledge is external
    if args.priorknowledge_type == "external" or args.priorknowledge_type == "random":
        gene_embeddings_location = os.path.join(args.src_path, args.gene_embeddings_path, args.priorknowledge_type, args.gene_embeddings)
        with open(gene_embeddings_location + '.npy', 'rb') as file:
            gene_embeddings = np.load(file, allow_pickle=True)

        gene_embeddings = torch.Tensor(gene_embeddings)
        print(f"[LOG] External Gene Embeddings: {gene_embeddings}")
        print(f"[LOG] External Gene Embeddings Shape: {gene_embeddings.size()}")

    ############################################## kfold ##############################################
    test_results_splits = {}
    i = 0
    for train_idx, val_idx, test_idx in zip(train_idxs, val_idxs, test_idxs):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]
        
        if not args.load_splits:
            # save patient ids to file
            np.save(save_dir + '/train_'+str(i)+'.npy', np.unique(train_df.patient_id) )
            np.save(save_dir + '/val_'+str(i)+'.npy', np.unique(val_df.patient_id) )
            np.save(save_dir + '/test_'+str(i)+'.npy', np.unique(test_df.patient_id) )


        # initialize datasets
        train_dataset = SuperTileRNADataset(train_df, args.feature_path)
        val_dataset = SuperTileRNADataset(val_df, args.feature_path)
        test_dataset = SuperTileRNADataset(test_df, args.feature_path)
        sampler = None
        shuffle = True

        train_dataloader = DataLoader(train_dataset, 
                    num_workers=0, pin_memory=True, 
                    shuffle=shuffle, batch_size=args.batch_size,
                    collate_fn=custom_collate_fn, sampler=sampler,
                    worker_init_fn=seed_worker,
                    generator=g)
        
        val_dataloader = DataLoader(val_dataset, 
                    num_workers=0, pin_memory=True, 
                    shuffle=True, batch_size=args.batch_size,
                    collate_fn=custom_collate_fn)
        
        test_dataloader = DataLoader(test_dataset, 
                    num_workers=0, pin_memory=True, 
                    shuffle=False, batch_size=args.batch_size,
                    collate_fn=custom_collate_fn)

        num_outputs = args.num_genes

        # extract different gene embeddings for each fold if priorknowledge type is internal
        if args.priorknowledge_type == "internal" or args.priorknowledge_type == "combined":
            gene_embeddings_location = os.path.join(args.src_path, args.gene_embeddings_path, args.priorknowledge_type, args.gene_embeddings + "_R"+ str(args.r_threshold) + "_fold" + str(i))
            with open(gene_embeddings_location + '.npy', 'rb') as file:
                gene_embeddings = np.load(file, allow_pickle=True)
                print("internal succeeded")
                print(gene_embeddings.shape)

            gene_embeddings = torch.Tensor(gene_embeddings)
            print(f"[LOG] Internal Gene Embeddings Fold {i}: {gene_embeddings}")
            print(f"[LOG] Internal Gene Embeddings Fold {i} Shape: {gene_embeddings.size()}")

        # load correct model architecture (type and version)
        if args.model_type == 'vis' and args.model_version == 'Regular':
            model = ViS(num_outputs=num_outputs, input_dim=args.feature_dim, depth=args.depth, nheads=args.num_heads, dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device) 
        elif args.model_type == 'vis' and args.model_version == 'Unbalanced':
            model = ViSPriorKnowledgeUnbalanced(num_outputs=num_outputs, input_dim=args.feature_dim, depth=args.depth, nheads=args.num_heads,  dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device, lam=args.lam, gene_embeddings=gene_embeddings)
        elif args.model_type == 'vis' and args.model_version == 'Balanced':
            model = ViSPriorKnowledgeBalanced(num_outputs=num_outputs, input_dim=args.feature_dim, depth=args.depth, nheads=args.num_heads,  dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device, lam=args.lam, gene_embeddings=gene_embeddings)
        else:
            print('please specify correct model type "vit" or "vis" and model version "Unbalanced", "Balanced"')
        print(f"[LOG] Model loaded: {args.model_type} in version {args.model_version}")
        
        if args.checkpoint and not args.change_num_genes:
            print("[LOG] Loading checkpoint from trained model...")
            suff = f'_{i}' if i > 0 else ''
            ck_path = os.path.join(args.src_path, args.save_dir, args.cohort, args.checkpoint, f'model_best{suff}.pt')
            model.load_state_dict(torch.load(ck_path, map_location = device))
        
        # sending model to device
        print("[LOG] sending model to device")
        model = model.to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()), lr=args.lr, amsgrad=False, weight_decay=0.)
        dataloaders = { 'train': train_dataloader, 'val': val_dataloader }
        
        # train model
        if args.train:
            print("[LOG] initiating training process")
            model = train(model, dataloaders, optimizer, save_dir=save_dir, run=run, split=i, save_on=args.save_on, stop_on=args.stop_on, delta=0.5)

        if not args.train:
            test_dataset = SuperTileRNADataset(df, args.feature_path)

            test_dataloader = DataLoader(test_dataset, 
                        num_workers=0, pin_memory=True, 
                        shuffle=False, batch_size=args.batch_size,
                        collate_fn=custom_collate_fn)

        preds, real, wsis, projs = evaluate(model, test_dataloader, run=run, suff='_'+str(i), dataset='test')

        # initialize a random model with same architecture (type and version)
        print("[LOG] initializing random model")
        if args.model_type == 'vis' and args.model_version == 'Regular':
            random_model = ViS(num_outputs=num_outputs, input_dim=args.feature_dim, depth=args.depth, nheads=args.num_heads, dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device)  
        elif args.model_type == 'vis' and args.model_version == 'Unbalanced':
            random_model = ViSPriorKnowledgeUnbalanced(num_outputs=num_outputs, input_dim=args.feature_dim, depth=args.depth, nheads=args.num_heads,  dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device, lam=args.lam, gene_embeddings=gene_embeddings)
        elif args.model_type == 'vis' and args.model_version == 'Balanced':
            random_model = ViSPriorKnowledgeBalanced(num_outputs=num_outputs, input_dim=args.feature_dim, depth=args.depth, nheads=args.num_heads,  dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device, lam=args.lam, gene_embeddings=gene_embeddings)
        random_model = random_model.to(device)
        random_preds, _, _, _ = evaluate(random_model, test_dataloader, run=run, suff='_'+str(i)+'_rand', dataset='test')

        # write results
        print("[LOG] writing results")
        test_results = {
            'real': real,
            'preds': preds,
            'random': random_preds,
            'wsi_file_name': wsis,
            'tcga_project': projs,
            'genes':[x for x in df.columns if 'rna_' in x]
        }

        test_results_splits[f'split_{i}'] = test_results
        i += 1
    
    test_results_splits['genes'] = [x[4:] for x in df.columns if 'rna_' in x]
    if args.train:
        with open(os.path.join(save_dir, 'test_results.pkl'), 'wb') as f:
            pickle.dump(test_results_splits, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("[LOG]: Printed eval results")
        with open(os.path.join(save_dir, 'test_results_CPTAC.pkl'), 'wb') as f:
            pickle.dump(test_results_splits, f, protocol=pickle.HIGHEST_PROTOCOL)

    wandb.finish()