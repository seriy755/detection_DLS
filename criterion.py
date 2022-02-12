import torch.optim as optim

class Criterion():
    def __init__(self, opt, params):
        opts = ['sgd', 'adam']
        if opt not in opts:
            print(f'{opt} is not correct, please enter correct criterion from: {opts}')
            raise NameError
        self.opt = opt
    
        self.lr = params['learning_rate']
        self.weight_decay = params['weight_decay']
        if opt == 'sgd':
            self.momentum = params['momentum']
        else:
            self.betas = params['betas']
    
    def __call__(self, model, lr=None):
        if lr is not None:
            self.lr = lr
        
        if self.opt == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, 
                                  weight_decay=self.weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.lr, betas = self.betas, 
                                  weight_decay=self.weight_decay)
        
        return optimizer
    
class Scheduler():
    def __init__(self, scheduler, params):
        schedulers = ['StepLR', 'MultiStepLR', 'ReduceLROnPlateau']
        if scheduler not in schedulers:
            print(f'{scheduler} is not correct, please enter correct criterion from: {schedulers}')
            raise NameError
        self.scheduler = scheduler
        
        if scheduler == 'StepLR':
            self.step_size = params['step_size']
            self.gamma = params['gamma']
            self.last_epoch = params['last_epoch']
        elif scheduler == 'MultiStepLR':
            self.milestones = params['milestones']
            self.gamma = params['gamma']
            self.last_epoch = params['last_epoch']
        else:
            self.mode = params['mode']
            self.factor = params['factor']
            self.patience = params['patience']
            self.verbose = params['verbose']
            self.threshold = params['threshold']
            self.threshold_mode = params['threshold_mode']
            self.cooldown = params['cooldown']
            self.min_lr = params['min_lr']
            self.eps = params['eps']
            
    def __call__(self, optimizer):
        if self.scheduler == 'StepLR':
            return optim.lr_scheduler.StepLR(optimizer, self,step_size, self.gamma, self.last_epoch)
        elif self.scheduler == 'MultiStepLR':
            return optim.lr_scheduler.MultiStepLR(optimizer, self.milestones, self.gamma, self.last_epoch)
        else:
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.mode, factor=self.factor,
                                                       patience = self.patience, verbose=self.verbose,
                                                       threshold=self.threshold, threshold_mode=self.threshold_mode,
                                                       cooldown=self.cooldown, min_lr=self.min_lr, eps=self.eps)