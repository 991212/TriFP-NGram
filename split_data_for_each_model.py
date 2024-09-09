import argparse
import pickle
from dataset import *
from torch_geometric.data import DataLoader
from config import hyperparameter
import os

def split_and_save_pickle_file_for_each_model(dataset, model_nums, cross_validation_folds):
    hp = hyperparameter()

    print("splitting and saving .pt file!!!")
    for cv_fold in range(cross_validation_folds):
        print('*' * 50, 'No_', cv_fold + 1, '_fold cross validation data', '*' * 50)
        data_root = 'dataset'
        data_folder = 'No_{}_fold cross validation data'.format(cv_fold+1)
        fpath = os.path.join(data_root, dataset, data_folder)

        '''split dataset to train set and test set'''
        train_set = GNNDataset(fpath, train=True)

        '''split dataset to train set and validation set'''
        train_val_folds = train_set.split_train_val_folds(model_nums,hp.seed)
        #保存每个模型的train和valid数据至每折的model(1-5)文件夹下
        for i_model, (train_data, valid_data) in enumerate(train_val_folds):
            
            model_data_dir = os.path.join(fpath, f'model{i_model + 1}')
            if not os.path.exists(os.path.join(fpath, f'model{i_model + 1}')):
                os.makedirs(os.path.join(fpath, f'model{i_model + 1}'))

            train_pkl_filename = os.path.join(fpath, f'model{i_model + 1}', f'train_data.pt')
            valid_pkl_filename = os.path.join(fpath, f'model{i_model + 1}', f'valid_data.pt')

            torch.save(train_data, train_pkl_filename)
            torch.save(valid_data, valid_pkl_filename)

            print(f"Model {i_model+1} in fold {cv_fold+1} train and validation data saved as .pt files.")



parser = argparse.ArgumentParser(prog='DTANet',description='Drug-protein Affinity',epilog='Model config set by config.py')

parser.add_argument('dataSetName', choices=["EC50", "IC50", "Ki", "Kd","kiba"],help='Enter which dataset to use for the experiment')
parser.add_argument('-n','--model_nums', type=int, default=5, help='Sets the number of ensemble models, the default is 5')
parser.add_argument('-cv_f','--cross_validation_folds', type=int, default=5, help='Set the number of folds for cross validation, the default is 5')

args = parser.parse_args()

split_and_save_pickle_file_for_each_model(dataset=args.dataSetName, model_nums=args.model_nums, cross_validation_folds=args.cross_validation_folds)