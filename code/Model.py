import ImageUtils
from ImageUtils import parse_record
import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from NetWork import ResNet

class Cifar(nn.Module):
    def __init__(self, config):
        super(Cifar, self).__init__()
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network = ResNet(
            self.config.resnet_size,
            self.config.first_num_filters,
            self.config.num_classes,
            self.device
        )
        # Define cross entropy loss and optimizer
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), momentum=0.9, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
    
    def train(self, x_train, y_train, max_epoch):
        assert self.config.resume_checkpoint >= 0
        if self.config.resume_checkpoint > 0:
            self.load(self.config.resume_checkpoint)
        self.network.train()
        # Determine how many batches in an epoch
        num_samples = x_train.shape[0]
        batch_size = self.config.initial_batch_size
        num_batches = num_samples // batch_size

        print('Learning rate = {}'.format(self.config.learning_rate))
        print('### Training... ###')
        for epoch in range(self.config.resume_checkpoint + 1, max_epoch+1):
            start_time = time.time()
            # Increase mini batch size
            if epoch > 60 and epoch % 10 == 1:
                print('Increasing batch size from {} to {}'.format(batch_size, 2 * batch_size))
                batch_size = 2 * batch_size
                num_batches = num_samples // batch_size
            # Shuffle
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            epoch_loss = 0
            for i in range(num_batches):
                # Parse mini batch
                x_train_batch = np.empty((batch_size, 3, 32, 32))
                for j in range(batch_size):
                    x_train_batch[j] = parse_record(x_train[i * batch_size + j], True)
                y_train_batch = y_train[i * batch_size : (i + 1) * batch_size]
                x_train_batch = torch.from_numpy(x_train_batch).float().to(self.device)
                y_train_batch = torch.from_numpy(y_train_batch).long().to(self.device)
                # Forward
                y_hat_batch = self.network(x_train_batch)
                loss = self.cross_entropy_loss(y_hat_batch, y_train_batch)
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)
            
            epoch_loss = epoch_loss / num_batches
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, epoch_loss, duration))

            if epoch % self.config.save_interval == 0:
                self.save(epoch, epoch_loss)

    def evaluate(self, x, y, evaluate_checkpoints):
        self.network.eval()
        print('### Test or Validation ... ###')
        for checkpoint_num in evaluate_checkpoints:
            self.load(checkpoint_num)
            preds = []
            for i in tqdm(range(x.shape[0])):
                x_i = torch.empty((1, 3, 32, 32), dtype=torch.float).to(self.device)
                x_i[0] = torch.from_numpy(parse_record(np.copy(x[i]), False))
                y_hat_i = self.network(x_i)
                preds.append(torch.argmax(y_hat_i, dim=1))
            y = torch.tensor(y)
            preds = torch.tensor(preds)
            accuracy = torch.sum(preds==y)/y.shape[0]
            print('Test accuracy: {:.4f}'.format(accuracy))

    def predict_probability(self, x, checkpoint):
        self.network.eval()
        print('### Probability Prediction ... ###')
        self.load(checkpoint)
        output = np.empty((x.shape[0], self.config.num_classes))
        for i in tqdm(range(x.shape[0])):
            x_i = torch.empty((1, 3, 32, 32), dtype=torch.float).to(self.device)
            x_i[0] = torch.from_numpy(parse_record(np.copy(x[i]), False))
            y_hat_i = self.network(x_i)
            y_hat_i = nn.Softmax(dim=1)(y_hat_i)
            assert(y_hat_i.shape == (1, 10))
            output[i] = y_hat_i[0].cpu().detach().numpy()
        np.save('../predictions.npy', output)

    def save(self, epoch, epoch_loss):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'model-' + str(epoch).zfill(4) + '.ckpt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch_loss': epoch_loss
            }, checkpoint_path)
        print("Checkpoint has been created.")
    
    def load(self, epoch):
        checkpoint_path = os.path.join(self.config.checkpoint_dir, 'model-' + str(epoch).zfill(4) + '.ckpt')
        checkpoint = torch.load(checkpoint_path)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        assert epoch == checkpoint['epoch']
        loss = checkpoint['epoch_loss']
        print("Restored model parameters for checkpoint {} loss {}".format(epoch, loss))