import os
import argparse
from tqdm import tqdm
import pickle

import h5py
from torch.utils.data import DataLoader, WeightedRandomSampler
import wandb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import torch.nn as nn
import random

from read_data import *
from utils import patient_kfold, grouped_strat_split, patient_split
from vit_new import train, ViT, evaluate, ViT_PriorKnowledge, PriorKnowledgeLinear, ViT_PriorKnowledgeUnbalanced

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
    projects = np.unique(df.tcga_project)
    all_wsis_with_features = []
    remove = []

    for proj in projects:
        wsis_with_features = os.listdir(feature_path + proj)
        # filter the ones without cluster_features
        for wsi in wsis_with_features:
            try:
                with h5py.File(feature_path +proj+ '/'+wsi+'/'+wsi+'.h5', "r") as f:
                    cols = list(f.keys())
                    if 'cluster_features' not in cols:
                        remove.append(wsi)
            except Exception as e:
                remove.append(wsi)
                
        all_wsis_with_features += wsis_with_features
    
    remove += df[~df['wsi_file_name'].isin(all_wsis_with_features)].wsi_file_name.values.tolist()
    print(f'Original shape: {df.shape}')
    df = df[~df['wsi_file_name'].isin(remove)].reset_index(drop=True)
    print(f'New shape: {df.shape}')

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
    parser.add_argument('--ref_file', type=str, default='examples/ref_file.csv', help='path to the reference csv file')
    parser.add_argument('--tcga_projects', help="the tcga_projects we want to use, separated by comma", default=None, type=str)
    parser.add_argument('--feature_path', type=str, default="examples/features/", help='path to resnet and clustered features')
    parser.add_argument('--save_dir', type=str, default='vit_exp', help='parent destination folder')
    parser.add_argument('--cohort', type=str, default="TCGA", help='cohort name for creating the saving folder of the results')
    parser.add_argument('--exp_name', type=str, default="exp", help='Experiment name for creating the saving folder of the results')
    parser.add_argument('--log', type=int, default=1, help='Whether to log the loss during training')

    # model args
    parser.add_argument('--seed', type=int, default=99, help='Seed for random generation')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint from trained model. If tcga_pretrain is true, then the model paths in different folds should be of the format args.checkpoint + "{fold}.pt" ')
    parser.add_argument('--train', help="if you want to train the model", action="store_true")
    parser.add_argument('--baseline', help="computing the baseline", action="store_true")
    parser.add_argument('--filter_genes', type=str, default=None, help='path to a npy file containing a list of genes of interest for training')
    parser.add_argument('--change_num_genes', help="whether finetuning from a model trained on different number of genes", action="store_true")
    parser.add_argument('--num_genes', type=int, default=25761, help='number of genes on which pretrained model was trained')
    parser.add_argument('--k', type=int, default=3, help='Number of splits')
    parser.add_argument('--tcga_pretrain', help="whether used pretrain model is pretrained on all TCGA cancers", action="store_true")
    parser.add_argument('--stratify', help="stratify k-fold for cancer type, only relevant if training on multiple cancer types", action="store_true")
    parser.add_argument('--balanced_sampling', help="balance sampling in dataloader according to number of available samples per cancer type, only relevant if training on multiple cancer types", action="store_true")
    parser.add_argument('--save_on', type=str, default='loss', help='which criterium to save model on, "loss" or "loss+corr"')
    parser.add_argument('--stop_on', type=str, default='loss', help='which criterium to do early stopping on, "loss" or "loss+corr"')
    parser.add_argument('--gene_embeddings', type=str, default='gene_embeddings/gene_embeddings_normalized_brca.npy', help='path to the file that contains the gene embeddings')
    parser.add_argument('--testing', type=int, default=0, help='decrease the amount of slides to run a quick test')
    

    ############################################## variables ##############################################
    args = parser.parse_args()

    ############################################## seeds ##############################################
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False # possibly reduced performance but better reproducibility
    torch.backends.cudnn.deterministic = True

    # reproducibility train dataloader
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(0)

    ############################################## logging ##############################################
    save_dir = os.path.join(args.src_path, args.save_dir, args.cohort, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run = None
    if args.log:
        print("trying to login wandb")
        wandb.login(key="3e7959a551c0e9b3c350df33b741771da9df8ea7") 

    ############################################## data prep ##############################################
    print("Starting data prep...")
    path_csv = args.ref_file
    df = pd.read_csv(path_csv)

    num_epochs = 200
    # testing purposes
    if args.testing:
        df = df.iloc[:20]
        num_epochs = 5

    # filter tcga projects
    if ('tcga_project' in df.columns) and (args.tcga_projects != None):
        projects = args.tcga_projects.split(',')
        df = df[df['tcga_project'].isin(projects)].reset_index(drop=True)

    print("filtering slides whose cluster features do not exist")
    df = filter_no_features(df, args.feature_path)

    # filter genes of interest
    selected_genes = None
    if args.filter_genes is not None:
        selected_genes = np.load(args.filter_genes, allow_pickle=True)
        selected_genes = list(set(selected_genes))
        df = filter_genes(df, selected_genes)
        print(f"Training only for selected genes: n = {len(selected_genes)}")

    ############################################## train, val, test split ##############################################
    print("Starting data split...")
    train_idx, val_idx, test_idx = patient_split(df)

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]

    # save patient ids to file
    np.save(save_dir + '/train.npy', np.unique(train_df.patient_id) )
    np.save(save_dir + '/val.npy', np.unique(val_df.patient_id) )
    np.save(save_dir + '/test.npy', np.unique(test_df.patient_id) )

    # creating the datasets
    train_dataset = SuperTileRNADataset(train_df, args.feature_path)
    val_dataset = SuperTileRNADataset(val_df, args.feature_path)
    test_dataset = SuperTileRNADataset(test_df, args.feature_path)

    if args.baseline:
        rna_columns = [x for x in train_df.columns if 'rna_' in x]
        rna_values = train_df[rna_columns].values
        mean_baseline = np.mean(rna_values, axis=0)

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
    
    # extract gene embeddings
    gene_embeddings_path = os.path.join(args.src_path, args.gene_embeddings)
    with open(gene_embeddings_path, 'rb') as file:
        gene_embeddings = np.load(file, allow_pickle=True)

    gene_embeddings = torch.Tensor(gene_embeddings.T)
    print(f"Gene Embeddings: {gene_embeddings}")

    # training from scratch
    num_outputs = train_dataset.num_genes

    # keep track of log values
    test_results_configs = {}

    for lam in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]:
        name = args.exp_name + "_" + str(lam)
        run = wandb.init(project="thesis", entity='maxhallemeesch', config=args, name=name)
        print(f"Using lambda value {lam}")
                        
        # create new model    
        model = ViT_PriorKnowledgeUnbalanced(num_outputs=args.num_genes, dim=2048, depth=6, heads=16, mlp_dim=2048, dim_head = 64, lam=lam, gene_embeddings=gene_embeddings)
            
        # load model if possible
        if args.checkpoint and not args.change_num_genes:
            model.load_state_dict(torch.load(args.checkpoint))
            
        model = model.cuda()

        optimizer = torch.optim.AdamW(list(model.parameters()), 
                                            lr=args.lr, 
                                            amsgrad=False,
                                            weight_decay=0.)

        dataloaders = {
            'train': train_dataloader,
            'val': val_dataloader
        }
            
        # train model
        if args.train:
            model = train(model, dataloaders, optimizer, save_dir=save_dir, run=run, save_on=args.save_on, stop_on=args.stop_on, delta=0.5, lam=lam, num_epochs=num_epochs)

        print("Printing Validation Results")
        preds, real, wsis, projs = evaluate(model, val_dataloader, run=run, dataset='val')

        #print("Printing Test Results") => should not be printed tbh
        #preds, real, wsis, projs = evaluate(model, test_dataloader, run=run, suff='_'+str(i))


        #random_model = ViT_PriorKnowledge(num_outputs=args.num_genes, dim=2048, depth=6, heads=16, mlp_dim=2048, dim_head = 64, lam=lam, gene_embeddings=gene_embeddings) 
        #random_model = random_model.cuda()
        #random_preds, _, _, _ = evaluate(random_model, val_dataloader, run=run, dataset='val', suff='_rand')

        test_results = {
            'real': real,
            'preds': preds,
            #'random': random_preds,
            'wsi_file_name': wsis,
            'tcga_project': projs,
            'genes':[x for x in df.columns if 'rna_' in x]
        }


        if args.baseline:
            mean_baseline = mean_baseline.reshape(-1, mean_baseline.shape[0])
            mean_baseline = np.repeat(mean_baseline, real.shape[0], axis=0)
            mse = mean_squared_error(real, mean_baseline)
            mae = mean_absolute_error(real, mean_baseline)
            print(f'Baseline test MSE {mse}')
            print(f'Baseline test MAE {mae}')
            test_results['baseline'] = mean_baseline

        test_results_configs[f'lambda_{lam}'] = test_results
        wandb.finish()

    test_results_configs['genes'] = [x[4:] for x in df.columns if 'rna_' in x]
    with open(os.path.join(save_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results_configs, f, protocol=pickle.HIGHEST_PROTOCOL)