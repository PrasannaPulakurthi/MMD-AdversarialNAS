import matplotlib.pyplot as plt
import numpy as np
import os

class PerformanceStore:
    def __init__(self, best_fid=1e4, best_is=0):
        self.fid = []
        self.inception_score = []
        self.best_fid = best_fid
        self.best_is = best_is
        self.best_epoch = -1 #?
        self.epochs = []
        self.init_fid = None
        self.init_is = None

    def update(self, fid, iscore, epoch):
        self.fid.append(fid)
        self.inception_score.append(iscore)
        self.epochs.append(epoch)

        if fid < self.best_fid:
            self.best_fid = fid
            self.best_epoch = epoch
            self.best_is = iscore
    def set_init(self, fid, iscore):
        self.init_fid = fid
        self.init_is = iscore
        
    def plot(self, path, plot_init=True):
        plt.figure()
        plt.plot(self.epochs, self.fid, label='FID')
        plt.plot(self.best_epoch, self.best_fid, 'ro', label='Best FID')
        try:
            if self.init_fid and plot_init:
                plt.plot(self.epochs, [self.init_fid]*len(self.epochs), '--k', label='Initial FID', alpha=0.5)
        except:
            pass
        plt.xlabel('Epoch')
        plt.ylabel('FID')
        plt.legend()
        plt.savefig(os.path.join(path, 'fid.png'))
        plt.close()

        plt.figure()
        plt.plot(self.epochs, self.inception_score, label='Inception Score')
        plt.plot(self.best_epoch, self.best_is, 'ro', label='Best Inception Score')
        try:
            if self.init_is and plot_init:
                plt.plot(self.epochs, [self.init_is]*len(self.epochs), '--k', label='Initial IS', alpha=0.5)
        except:
            pass
        plt.xlabel('Epoch')
        plt.ylabel('Inception Score')
        plt.legend()
        plt.savefig(os.path.join(path,'is.png'))
        plt.close()
    