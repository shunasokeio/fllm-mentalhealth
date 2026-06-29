import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import Subset, ConcatDataset
from torchvision import datasets

import numpy as np
import pickle

import sys

class DataSplitter:
    def __init__(self, args, dataset):
        np.random.seed(123)

        self.args               = args
        self.non_iid            = args.non_iid
        self.num_client         = args.num_client
        self.dataset            = dataset
        self.trainset           = dataset.trainset
        self.testset            = dataset.testset
        self.num_classes        = args.num_classes
        self.is_balanced        = args.is_balanced

        # Amount of data per class label
        # e.g.) CIFAR10: 5000(Train) / 1000(Test)
        self.train_class_size   = len(self.trainset)    // self.num_classes
        self.test_class_size    = len(self.testset)     // self.num_classes
        # Amount of data per client (Sharded data)
        # e.g.) CIFAR10, 100 clients: 500(Train) / 100(Test)
        self.train_shard_size_client   = self.train_class_size // self.num_client
        self.test_shard_size_client    = self.test_class_size  // self.num_client
        # Amount of data per client per class if iid
        # e.g.) CIFAR10, 100 clients: 50,50,50,50,50,50,50,50,50,50(Train per class)
        self.train_shard_size_client_class   = self.train_shard_size_client // self.num_classes
        self.test_shard_size_client_class    = self.test_shard_size_client  // self.num_classes

        # Indices per class label
        if hasattr(self.trainset, "targets"):
            self.train_labels = np.array(self.trainset.targets)
            self.test_labels = np.array(self.testset.targets)
        if hasattr(self.testset, "labels"):
            self.train_labels = np.array(self.trainset.labels)
            self.test_labels = np.array(self.testset.labels)

        # Dataset indices per class label
        self.train_class_indices = [np.where(self.train_labels == i)[0] for i in range(self.num_classes)]
        self.test_class_indices  = [np.where(self.test_labels == i)[0] for i in range(self.num_classes)]
        
        # Final Output
        # Sharded dataset per client
        self.sharded_trainset   = []
        self.sharded_testset    = []

        if self.non_iid == 0:
            self.is_iid = "strict_noniid"
        elif self.non_iid == sys.maxsize:
            self.is_iid = "iid"
        else:
            self.is_iid = "noniid"

    # Dirichlet non-iid degree = INF
    def split_iid(self):
        self.sharded_trainset = [[] for n in range(self.num_client)]
        self.sharded_testset = [[] for n in range(self.num_client)]

        for indices_per_class in self.train_class_indices:
            np.random.shuffle(indices_per_class)
            train_shard_indices = np.array_split(indices_per_class, self.num_client)
            for client_idx, shard in enumerate(train_shard_indices):
                self.sharded_trainset[client_idx].append(Subset(self.trainset, shard.tolist()))

        for indices_per_class in self.test_class_indices:
            np.random.shuffle(indices_per_class)
            test_shard_indices = np.array_split(indices_per_class, self.num_client)
            for client_idx, shard in enumerate(test_shard_indices):
                self.sharded_testset[client_idx].append(Subset(self.testset, shard.tolist()))

        for client_idx in range(self.num_client):
            self.sharded_trainset[client_idx] = ConcatDataset(self.sharded_trainset[client_idx])
            self.sharded_testset[client_idx] = ConcatDataset(self.sharded_testset[client_idx])   

    # Dirichlet non-iid degree > 0, Unbalanced local dataset size
    # Assume total number of local datasets == total number of dataset
    # i.e. Class 0: [0.1, 0.2, 0.01, ..., (X100)] -> "Per Class"
    #      Client 0: [0.1, 0.2, 0.01, ..., (X10)] -> "Per Client"
    def split_noniid_unbalanced(self):
        self.sharded_trainset = [[] for n in range(self.num_client)]
        self.sharded_testset = [[] for n in range(self.num_client)]

        # For each class, set random ratio for each client
        class_priors = np.random.dirichlet([self.non_iid] * self.num_client, self.num_classes)

        for class_idx, (train_indices, test_indices) in enumerate(zip(self.train_class_indices, self.test_class_indices)):
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

            # Largest Remainder Method
            train_alloc_float = class_priors[class_idx] * self.train_class_size
            train_alloc_int = np.floor(train_alloc_float).astype(int)
            train_remainder = self.train_class_size - np.sum(train_alloc_int)
            # Prevent last client from receiving too many leftovers
            if train_remainder > 0:
                # Distribute leftover from the client with the largest diff (float vs floored int)
                sorted_client_indices = np.argsort(-(train_alloc_float - train_alloc_int))
                for i in range(train_remainder):
                    train_alloc_int[sorted_client_indices[i]] += 1
            train_alloc = train_alloc_int
            train_split_indices = np.split(train_indices, np.cumsum(train_alloc)[:-1])

            test_alloc_float = class_priors[class_idx] * self.test_class_size
            test_alloc_int = np.floor(test_alloc_float).astype(int)
            test_remainder = self.test_class_size - np.sum(test_alloc_int)
            # Prevent last client from receiving too many leftovers
            if test_remainder > 0:
                # Distribute leftover from the client with the largest diff (float vs floored int)
                sorted_client_indices = np.argsort(-(test_alloc_float - test_alloc_int))
                for i in range(test_remainder):
                    test_alloc_int[sorted_client_indices[i]] += 1
            test_alloc = test_alloc_int
            test_split_indices = np.split(test_indices, np.cumsum(test_alloc)[:-1])

            for client_idx in range(self.num_client):
                if train_split_indices[client_idx].size > 0:
                    self.sharded_trainset[client_idx].append(Subset(self.trainset, train_split_indices[client_idx].tolist()))
                if test_split_indices[client_idx].size > 0:
                    self.sharded_testset[client_idx].append(Subset(self.testset, test_split_indices[client_idx].tolist()))

        # Ensure minimum number of local dataset to be at least one (Duplicate possible)
        for client_idx in range(self.num_client):
            if len(self.sharded_trainset[client_idx]) == 0:
                random_idx = np.random.choice(len(self.trainset))
                self.sharded_trainset[client_idx].append(Subset(self.trainset, [random_idx]))
            if len(self.sharded_testset[client_idx]) == 0:
                random_idx = np.random.choice(len(self.testset))
                self.sharded_testset[client_idx].append(Subset(self.testset, [random_idx]))

        # Concatenate subsets per client
        for client_idx in range(self.num_client):
            self.sharded_trainset[client_idx] = ConcatDataset(self.sharded_trainset[client_idx])
            self.sharded_testset[client_idx] = ConcatDataset(self.sharded_testset[client_idx])

    # Dirichlet non-iid degree > 0, Balanced local dataset
    # Assume total number of local datasets != total number of dataset
    def split_noniid_balanced(self):
        self.sharded_trainset   = [[] for n in range(self.num_client)]
        self.sharded_testset    = [[] for n in range(self.num_client)]

        client_priors = np.random.dirichlet([self.non_iid] * self.num_classes, self.num_client)

        train_alloc = client_priors * self.train_shard_size_client 
        test_alloc = client_priors * self.test_shard_size_client
        train_alloc, test_alloc = train_alloc.astype(int), test_alloc.astype(int)

        # Iterative Proportional Fitting 
        # Satisfy at best
        # 1) All client local dataset size to be same (i.e. CIFAR10 100 Clients: 500)
        # 2) All dataset size per class to be same (i.e. CIFAR10: 5000)
        max_iter = 1000
        for iter in range(max_iter):
            row_sums = train_alloc.sum(axis=1, keepdims=True)
            train_alloc = train_alloc * (self.train_shard_size_client / row_sums)
            train_alloc = train_alloc.astype(int)
            
            col_sums = train_alloc.sum(axis=0, keepdims=True)
            train_alloc = train_alloc * (self.train_class_size / col_sums)
            train_alloc = train_alloc.astype(int)

            if train_alloc.sum(axis=0, keepdims=True).all() == self.train_class_size and train_alloc.sum(axis=1, keepdims=True).all() == self.train_shard_size_client:
                break

        for iter in range(max_iter):
            row_sums = test_alloc.sum(axis=1, keepdims=True)
            test_alloc = test_alloc * (self.test_shard_size_client / row_sums)
            test_alloc = test_alloc.astype(int)
            
            col_sums = test_alloc.sum(axis=0, keepdims=True)
            test_alloc = test_alloc * (self.test_class_size / col_sums)
            test_alloc = test_alloc.astype(int)

            if test_alloc.sum(axis=0, keepdims=True).all() == self.test_class_size and test_alloc.sum(axis=1, keepdims=True).all() == self.test_shard_size_client:
                break

        train_alloc, test_alloc = np.transpose(train_alloc), np.transpose(test_alloc)

        for class_idx, (train_indices, test_indices) in enumerate(zip(self.train_class_indices, self.test_class_indices)):
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)
            train_split_indices = np.split(train_indices, np.cumsum(train_alloc[class_idx])[:-1])
            test_split_indices = np.split(test_indices, np.cumsum(test_alloc[class_idx])[:-1])
            for client_idx in range(self.num_client):
                if train_split_indices[client_idx].size > 0:
                    self.sharded_trainset[client_idx].append(Subset(self.trainset, train_split_indices[client_idx].tolist()))
                if test_split_indices[client_idx].size > 0:
                    self.sharded_testset[client_idx].append(Subset(self.testset, test_split_indices[client_idx].tolist()))

        # Ensure minimum number of local dataset to be at least one (Duplicate possible)
        for client_idx in range(self.num_client):
            if len(self.sharded_trainset[client_idx]) == 0:
                random_idx = np.random.choice(len(self.trainset))
                self.sharded_trainset[client_idx].append(Subset(self.trainset, [random_idx]))
            if len(self.sharded_testset[client_idx]) == 0:
                random_idx = np.random.choice(len(self.testset))
                self.sharded_testset[client_idx].append(Subset(self.testset, [random_idx]))

        for client_idx in range(self.num_client):
            self.sharded_trainset[client_idx] = ConcatDataset(self.sharded_trainset[client_idx])
            self.sharded_testset[client_idx] = ConcatDataset(self.sharded_testset[client_idx])

    # Dirichlet non-iid degree = 0 (Pathological)
    def split_strict_noniid(self):
        for i in range(self.num_client):                
            self.sharded_trainset.append(Subset(self.trainset, [n for n in range(i * self.train_shard_size, (i+1) * self.train_shard_size)]))
            self.sharded_testset.append(Subset(self.testset, [n for n in range(i * self.test_shard_size, (i+1) * self.test_shard_size)]))

    def shard_dataset(self, iid=False):
        if iid:
            self.is_iid = "iid"

        if self.is_iid == "iid":
            self.split_iid()
        if self.is_iid == "noniid":
            if self.is_balanced:
                self.split_noniid_balanced()
            else:
                self.split_noniid_unbalanced()
        if self.is_iid == "strict_noniid":
            self.split_strict_noniid()
        
        return self.sharded_trainset, self.sharded_testset