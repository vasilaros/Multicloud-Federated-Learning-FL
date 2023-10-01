import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util.utils import get_index_from_one_hot_label, get_even_odd_from_one_hot_label

def get_data(dataset, total_data, dataset_file_path=os.path.dirname(__file__), sim_round=None):

    if dataset == 'MNIST_ORIG_EVEN_ODD' or dataset == 'MNIST_ORIG_ALL_LABELS':
        from data_reader.mnist_extractor import mnist_extract

        if total_data > 60000:
            total_data_train = 60000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data

        if sim_round is None:
            start_index_train = 0
            start_index_test = 0
        else:
            start_index_train = (sim_round * total_data_train) % (max(1, 60000 - total_data_train + 1))
            start_index_test = (sim_round * total_data_test) % (max(1, 10000 - total_data_test + 1))

        train_image, train_label = mnist_extract(start_index_train, total_data_train, True, dataset_file_path)
        test_image, test_label = mnist_extract(start_index_test, total_data_test, False, dataset_file_path)

        # train_label_orig must be determined before the values in train_label are overwritten below
        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label[0])

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(train_label)):
                train_label[i] = get_even_odd_from_one_hot_label(train_label[i])

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(test_label)):
                test_label[i] = get_even_odd_from_one_hot_label(test_label[i])

    elif dataset == 'CIFAR_10':
        from data_reader.cifar_10_extractor import cifar_10_extract

        if total_data > 50000:
            total_data_train = 50000
        else:
            total_data_train = total_data

        if total_data > 10000:
            total_data_test = 10000
        else:
            total_data_test = total_data

        train_image, train_label = cifar_10_extract(0, total_data_train, True, dataset_file_path)
        test_image, test_label = cifar_10_extract(0, total_data_test, False, dataset_file_path)

        train_label_orig=[]
        for i in range(0, len(train_label)):
            label = get_index_from_one_hot_label(train_label[i])
            train_label_orig.append(label[0])
            
    elif dataset == 'LINEAR':
        print('Reading Linear dataset')
        from data_reader.linear_extractor import linear_extract
        X_train0, Y_train = linear_extract(is_train=True)
        X_test0, Y_test = linear_extract(is_train=False)

        X_train0 = np.array(X_train0)
        Y_train = np.array(Y_train) 
        X_test0 = np.array(X_test0)
        Y_test = np.array(Y_test)
    
        X_train = np.column_stack((X_train0, np.ones(X_train0.shape[0])))
        X_test = np.column_stack((X_test0, np.ones(X_test0.shape[0])))
        
        train_image = X_train
        train_label = Y_train
        test_image = X_test
        test_label = Y_test
        train_label_orig = Y_train
         
    elif dataset.split('.')[0] == 'LSTM':
        from data_reader.lstm_extractor import lstm_extract
        fname = dataset.split('.')[-1]
        print(f'Reading LSTM dataset: {fname}')
        n_train = 1500
        n_test = 500
        time_dimension = 60
        data = lstm_extract(fname)

        x_train = data[:time_dimension]
        y_train = data[time_dimension]
        for i in range(time_dimension+1, n_train):
            x_train = np.vstack((x_train, data[i-time_dimension:i]))
            y_train = np.vstack((y_train, data[i]))
        
        #train_image = x_train[:,:, np.newaxis]
        train_image = x_train[:,:, np.newaxis]
        train_label = y_train

        x_test = data[n_train-time_dimension:n_train]
        y_test = data[n_train]
        for i in range(n_train+1, n_train+n_test):
            x_test = np.vstack((x_test, data[i-time_dimension:i]))
            y_test = np.vstack((y_test, data[i]))
        
        #test_image = x_test[:,:, np.newaxis]
        test_image = x_test[:,:, np.newaxis]
        test_label = y_test
        train_label_orig = y_train
  
    else:
        raise Exception('Unknown dataset name.')

    return train_image, train_label, test_image, test_label, train_label_orig


def get_data_train_samples(dataset, samples_list, dataset_file_path=os.path.dirname(__file__)):
    if dataset == 'MNIST_ORIG_EVEN_ODD' or dataset == 'MNIST_ORIG_ALL_LABELS':
        from data_reader.mnist_extractor import mnist_extract_samples

        train_image, train_label = mnist_extract_samples(samples_list, True, dataset_file_path)

        if dataset == 'MNIST_ORIG_EVEN_ODD':
            for i in range(0, len(train_label)):
                train_label[i] = get_even_odd_from_one_hot_label(train_label[i])

    elif dataset == 'CIFAR_10':
        from data_reader.cifar_10_extractor import cifar_10_extract_samples

        train_image, train_label = cifar_10_extract_samples(samples_list, True, dataset_file_path)

    else:
        raise Exception('Training data sampling not supported for the given dataset name, use entire dataset by setting batch_size = total_data, ' +
                        'also confirm that dataset name is correct.')

    return train_image, train_label
