from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import os
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from fedlab.contrib.algorithm.basic_server import SyncServerHandler
from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.utils.functional import setup_seed, AverageMeter, evaluate


from settings import parse_args, get_settings, get_logs, get_heterogeneity
from utils import layerwise_model, fg, plot_to_tensorboard
from sampling import FeedbackSampler, UniformSampler
from aggregator import Aggregators
from utils import gradient_diversity


##################
#
#      Client
#
##################

class FedASTWSerialClientTrainer(SGDSerialClientTrainer):
    """Federated client with local SGD solver."""
    @ property
    def model_layerwise_parameters(self) -> List[torch.Tensor]:
        """Return serialized model parameters by layer.
        """
        return layerwise_model(self._model)
    
    def setup_optim(self, epochs, batch_size, lr, args=None):
        self.args = args
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr, weight_decay=5e-4)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.layer = len(self.model_layerwise_parameters)
        self.thresh = args.thresh # threshold for adaptive frequency

    def local_process(self, payload, id_list, t):
        """
        Client trains its local model on local dataset.
        Args:
            payload (List[torch.Tensor]): Serialized model parameters.
        """
        model_parameters = payload[0]
        self.thresh = payload[1]
        loss_ = AverageMeter()
        acc_ = AverageMeter()

        for id in tqdm(id_list):
            self.batch_size, self.epochs = get_heterogeneity(args, args.datasize_list[id])
            train_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, train_loader, loss_, acc_, t)
            self.cache.append(pack)
        
        
        return loss_, acc_
  
    def train(self, model_parameters, train_loader, loss_, acc_, curr_round): 
        """Train the model on local data.
        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
            train_loader (DataLoader): DataLoader for local dataset.
            loss_ (AverageMeter): AverageMeter for loss.
            acc_ (AverageMeter): AverageMeter for accuracy.
            curr_round (int): Current round number.
        
        Returns:
            List[List[torch.Tensor],List[bool]]:Layerwise model parameters and flag for one client.
        """

        self.set_model(model_parameters)
        frz_layerwise_parameters = deepcopy(self.model_layerwise_parameters) 

        for _ in range(self.epochs):
            self.model.train()
            for data,target in train_loader:
                batch_size = len(target)
                if self.cuda:
                    data, target = data.to(self.device), target.to(self.device)

                output = self._model(data)
                loss = self.criterion(output, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(output.data, 1) # get the index of the max log-probability, tensor(batch_size,)
                loss_.update(loss.item(), batch_size) # loss per batch
                acc_.update(torch.sum(predicted.eq(target)).item() / batch_size, batch_size)
        
        # get the gradient and norms of all layers, e.g. [layer1, layer2, ...], [norm1, norm2, ...]
        layerwise_update = [origin-curr for curr, origin in zip(self.model_layerwise_parameters, frz_layerwise_parameters)]
        layerwise_norm = [torch.norm(update,p=2).item() for update in layerwise_update]
        
        # set flag for each layer of model
        layer_flag = self.set_flag(curr_round, layerwise_norm if self.args.mode == 2 else None)
        
        # choose the active layer of model
        if self.args.mode != 0:
            layerwise_update =[layerwise_update[i] for i in range(self.layer) if layer_flag[i] == True]

        return (layerwise_update, layer_flag, layerwise_norm)
    
    def set_flag(self, round, norms=None):
        """Set flag for each layer of model.
        Args:
            mode (int): Mode of computation
             0: layer_sync; 
             1: layer_async with constant freq; 
             2: layer_async with adaptive freq.
            round (int): Current round number.
            norms (List[float]): Norms of gradients for each layer.
        Returns:
            flag (List[bool]): Flag for each layer of model.
        """
        flag = [True]*self.layer # flag:List[bool], length:layer
        if self.args.mode == 1:
            if round % args.b not in set(range(0, self.args.a)):
                # update only shallow layer
                mid = self.layer - 2
                for i in range(mid, self.layer):
                    flag[i] = False
        elif self.args.mode == 2:
            if norms is not None:
                assert len(norms) == len(flag), "norms and flag should have the same length."
                for i in range(len(norms)):
                    flag[i] =  False if norms[i] < self.thresh else True
            else:
                raise ValueError("norms should not be None when mode is 2.")
        assert len(flag) == self.layer, "flag and layer should have the same length."
        return flag
    


##################
#
#      Server
#
##################

class Server_ASTW_GradientCache(SyncServerHandler):
    """Server for FedASTW.
    Args:
        model (nn.Module): Model to be trained.
        global_round (int): Number of rounds for training.
        sample_ratio (float): Ratio of clients to be sampled in each round.
    """
    @property
    def model_layerwise_parameters(self) -> List[torch.Tensor]:
        """Return serialized model parameters by layer.
        """
        return layerwise_model(self._model)
    

    def setup_optim(self, sampler, args):  
        """Setup optimization configuration.
        Args:
            self.round_clients (int): Number of clients sampled in each round.
            self.sampler (Sampler): Sampler for clients.
            self.lr (float): global Learning rate.
            self.layer (int): Number of layers in the model.
            self.timestamp (List[List[int]]): Timestamp for each layer and client,(layer, num_clients).
            self.momentum (List[List[torch.Tensor]]): Momentum for each layer and client,(layer, num_clients,).
        """
        self.round_clients = max(1,int(self.sample_ratio*self.num_clients))
        self.sampler = sampler
        self.args = args
        self.lr = args.glr
        self.layer = len(self.model_layerwise_parameters)
        self.timestamp = [[0]*self.layer for _ in range(self.num_clients)] # timestamp:List[List[int]],(num_clients, layer)
        self.momentum  = [[torch.zeros_like(param) for param in self.model_layerwise_parameters] 
                    for _ in range(self.num_clients)] # momentum:List[List[torch.Tensor]],(num_clients, layer)
        self.alpha = args.alpha # momentum coefficient
        self.gnorm = [0]*self.layer # gnorm:List[float], length:layer, pseudo-gradient for global gradient descent

    def timestamp_update(self, buffer):
        """Update timestamp for each layer and client.
        """
        indices, _ = self.sampler.last_sampled
        for idx in range(self.round_clients):
            for layer_idx in range(self.layer):
                if buffer[idx][1][layer_idx] == True:
                    self.timestamp[indices[idx]][layer_idx] = self.round
                else:
                    self.timestamp[indices[idx]][layer_idx] = max(0,self.round - 1)


    def momentum_update(self, buffer):
        """Update momentum for each layer and client.
        """
        indices, _ = self.sampler.last_sampled
        for idx in range(self.round_clients):
            j = 0 # pointer for current layer of each client
            for layer_idx in range(self.layer):
                if buffer[idx][1][layer_idx] == True:
                    self.momentum[indices[idx]][layer_idx] = (1-self.alpha) * self.momentum[indices[idx]][layer_idx] + self.alpha * buffer[idx][0][j]
                    j += 1
                else:
                    self.momentum[indices[idx]][layer_idx] = (1-self.alpha) * self.momentum[indices[idx]][layer_idx] 


    def compute_weights(self, tw, all_clients=True):
        """Compute weights for each layer and client.
        Args:
            tw(bool): Whether to use time-weighted sampling.
            all_clients(bool): Whether to use all clients or only sampled clients.
        
        Returns:
            weights:List[List[float]], length: num_clients, weights for each layer and client.
        """
        indices, _ = self.sampler.last_sampled
        weights = [[x] * self.layer for x in self.args.datasize_list] 
        if tw:
            for client_idx in indices: 
                for layer_idx in range(self.layer):
                    weights[client_idx][layer_idx] *= fg(self.timestamp[client_idx][layer_idx], self.round, self.args.tw_a)
        
        if not all_clients:
            weights = [weights[i] for i in indices] 

        # normalize the weights by layer
        for i in range(self.layer):
            layer_weight = [weight[i] for weight in weights]
            # check if the weights are non-negative
            if any(w < 0 for w in layer_weight):
                raise ValueError(f"Layer {i} weights must be non-negative.")
            # normalize the weights
            layer_weight = np.array(layer_weight)/ np.sum(layer_weight) 
            
            for client_idx in range(len(weights)):
                weights[client_idx][i] = layer_weight[client_idx]

        return weights

        
    @property
    def num_clients_per_round(self):
        return self.round_clients
           
    def sample_clients(self, k, startup=0):
        """Sample clients for the current round.
        Args:
            k (int): Number of clients to be sampled.
            startup (int): Whether to use startup sampling.
        Returns:
            clients (np.array): Sampled clients.
        """
        if self.sampler.name == "uniform":
            clients = self.sampler.sample(k)
        elif self.sampler.name == "feedback":
            clients = self.sampler.sample(k, startup)
        else:
            raise ValueError("Invalid sampler type.")
        
        self.round_clients = len(clients)
        assert self.num_clients_per_round == len(clients)
        return clients
    
    @property
    def downlink_package(self) -> Tuple[torch.Tensor,List[float]]:
        """Property for manager layer. Server manager will call this property when activates clients."""
        return [self.model_parameters, self.thresh] # model_parameters:torch.Tensor, thresh: List[float]
    
    def global_update(self, buffer):
        """Update global model with collected parameters from clients.
        Args:
            buffer List[Tuple[List[torch.Tensor],List[bool],List[float]]]): Serialized model parameters, layer flag, and norms.
                buffer[idx][0]: List[torch.Tensor], length: active layer.
                buffer[idx][1]: List[bool], length: layer.
                buffer[idx][2]: List[float], length: layer.
        """
        self.momentum_update(buffer) # update momentum
        if self.args.tw:
            self.timestamp_update(buffer) # update timestamp
        indices, _ = self.sampler.last_sampled
            
        if self.sampler.name == "feedback":
            # feedback sampling
            if self.sampler.explored:
                # aggregate all clients
                weights = self.compute_weights(self.args.tw) 
                estimates = Aggregators.fedastw_aggregate(self.momentum, weights, self.layer)
                probs_value = [sum(weight) for weight in weights]
                # absolute difference between probs_value and self.layer
                assert abs(sum(probs_value) - self.layer) < 1e-5, "probs_value should be equal to the number of layers."
                probs_value = np.array(probs_value) / np.sum(probs_value) # normalize the probs_value
                self.sampler.update(probs_value,beta=1)
            
            else:  # explore phase
                # aggregate only sampled clients
                weights = self.compute_weights(self.args.tw, all_clients=False) 
                active_momentum = [self.momentum[i] for i in indices]
                estimates = Aggregators.fedastw_aggregate(active_momentum, weights, self.layer)

        elif self.sampler.name == "uniform":
            # uniform sampling
            if self.args.all_clients:
                weights = self.compute_weights(self.args.tw) 
                estimates = Aggregators.fedastw_aggregate(self.momentum, weights, self.layer)
            else:
                weights = self.compute_weights(self.args.tw, all_clients=False) 
                active_momentum = [self.momentum[i] for i in indices]
                estimates = Aggregators.fedastw_aggregate(active_momentum, weights, self.layer)
        else:
            raise ValueError("Invalid sampler type.")
        
        serialized_estimates = torch.cat(estimates, dim=0)
        self.gnorm = [torch.norm(gradient, p=2).item() for gradient in estimates] # gnorm:List[float], length:layer
        serialized_parameters = self.model_parameters - self.lr*serialized_estimates
        self.set_model(serialized_parameters)

    
    def load(self, payload:Tuple[List[torch.Tensor],List[bool],List[float]]) -> bool:
        """Update global model with collected parameters from clients.

        Note:
            Server handler will call this method when its ``client_buffer_cache`` is full. User can
            overwrite the strategy of aggregation to apply on :attr:`model_parameters_list`, and
            use :meth:`SerializationTool.deserialize_model` to load serialized parameters after
            aggregation into :attr:`self._model`.

        Args:
            payload (Tuple[List[torch.Tensor],List[bool],List[float]]): Serialized model parameters, layer flag, and norms.
                payload[0]: List[torch.Tensor], length: active layer.
                payload[1]: List[bool], length: layer.
                payload[2]: List[float], length: layer.
        """
        assert len(payload) == 3, "payload should be a tuple of (model_parameters, layer_flag, layer_norms)."
        self.client_buffer_cache.append(deepcopy(payload))

        assert len(self.client_buffer_cache) <= self.num_clients_per_round

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1

            # reset cache
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False


#####################
#                   #
#      Pipeline     #
#                   #
#####################

if __name__ == "__main__":
    args = parse_args()
    args.method = "fedastw"
    args.K = max(1,int(args.num_clients*args.sample_ratio)) # number of active clients each round
    setup_seed(args.seed)

    path = get_logs(args)
    writer = SummaryWriter(path)
    print(f"Running Log saved in {path}")
    json.dump(vars(args), open(os.path.join(path, "config.json"), "w"))

    model, dataset, datasize_list, gen_test_loader = get_settings(args)
    args.datasize_list = datasize_list

    # client-trainer setup
    trainer = FedASTWSerialClientTrainer(model,args.num_clients,cuda=True)
    trainer.setup_optim(args.epochs, args.batch_size, args.lr, args)
    trainer.setup_dataset(dataset)

    # server-handler setup
    handler = Server_ASTW_GradientCache(model=model,
                                        global_round=args.com_round,
                                        sample_ratio=args.sample_ratio)
    handler.num_clients = trainer.num_clients

    # sampler setup
    probs = np.ones(args.num_clients)/args.num_clients
    if args.sampler_name == "uniform":
        sampler = UniformSampler(args.num_clients, probs)
    elif args.sampler_name == "feedback":
        # feedback sampling
        sampler = FeedbackSampler(args.num_clients, probs)
    else:
        raise ValueError("Invalid sampler type.")
    handler.setup_optim(sampler, args)

    local_update_norm_track = []


    t = 0 # track the current round number
    while handler.if_stop is False:
        ## server side ##
        # sample clients
        if t == 0 :
            sampled_clients = handler.sample_clients(args.K, args.startup) # sorted clients idx# sampled_clients: np.array
        else:
            sampled_clients = handler.sample_clients(args.K)
        broadcast = handler.downlink_package # List[torch.Tensor]

        ## client side ##
        train_loss, train_acc = trainer.local_process(broadcast, sampled_clients,t)
        uploads = trainer.uplink_package # List[Tuple[List[torch.Tensor],List[bool],List[float]]],length: round_clients
        
        ## log local update norm ##
        # local update norm of all layers
        local_update_norm = [np.linalg.norm(np.array(uploads[idx][2])) for idx in range(len(uploads))]  # List[float], length: round_clients
        writer.add_scalar('Metric/LocalUpdateNorm/{}'.format(args.dataset), np.mean(local_update_norm), t)
        writer.add_scalar('Metric/StdLocalUpdateNorm/{}'.format(args.dataset), np.std(local_update_norm), t)
        local_update_norm_track.append(local_update_norm)

        # local update norm by layer
        layer = trainer.layer
        layer_gradient_list = []
        p = [0] * len(uploads) # p:List[int], length: num_clients, pointer for current layer of each client
        for i in range(layer): 
            # layer i
            for idx in range(len(uploads)):
                # client idx
                if uploads[idx][1][i] == True:
                    layer_gradient_list.append(uploads[idx][0][p[idx]]) # NB: only including uploaded gradients
                    p[idx] += 1
            
            layer_norms_list = [uploads[idx][2][i] for idx in range(len(uploads))] # including all sampled clients
            writer.add_scalar('Metric/layer{}/MeanNorm/{}'.format(i+1,args.dataset), np.mean(layer_norms_list), t)
            writer.add_scalar('Metric/layer{}/MaxNorm/{}'.format(i+1,args.dataset), np.max(layer_norms_list), t)
            writer.add_scalar('Metric/layer{}/MinNorm/{}'.format(i+1,args.dataset), np.min(layer_norms_list), t)
            writer.add_scalar('Metric/layer{}/StdNorm/{}'.format(i+1,args.dataset), np.std(layer_norms_list), t)
            
            diversity = gradient_diversity(layer_gradient_list)
            writer.add_scalar('Metric/layer{}/Diversity/{}'.format(i+1,args.dataset), diversity, t)
            

        ## server side ##
        for pack in uploads: # pack: Tuple[List[torch.Tensor],List[bool],List[float]]
            handler.load(pack)
        
        # log global pseudo-gradient norm
        global_norm = handler.gnorm  # global_norm:List[float], length:layer
        assert len(global_norm) == handler.layer, "global_norm and layer should have the same length."
        # log global pseudo-gradient norm by layer
        for i in range(layer):
            writer.add_scalar('Metric/GlobalNorm/layer{}/{}'.format(i+1,args.dataset), global_norm[i], t)
        # log global pseudo-gradient norm of all layers
        writer.add_scalar('Metric/GlobalNorm/{}'.format(args.dataset), np.linalg.norm(np.array(global_norm)), t)

        if t%args.freq == 0:  # frequency of evaluation
            test_loss, test_acc = evaluate(handler._model, nn.CrossEntropyLoss(), gen_test_loader) 
            
            # log test loss and accuracy
            writer.add_scalar('Train/loss/{}'.format(args.dataset), train_loss.avg, t)
            writer.add_scalar('Train/accuracy/{}'.format(args.dataset), train_acc.avg, t)
            writer.add_scalar('Test/loss/{}'.format(args.dataset), test_loss, t)
            writer.add_scalar('Test/accuracy/{}'.format(args.dataset), test_acc, t)

            print(f"Round {t}: Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc.avg:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

        t += 1
    
    # save figure to tensorboard
    assert len(local_update_norm_track) == args.com_round, "local_update_norm_track and com_round should have the same length."
    plot_to_tensorboard(np.arange(args.com_round), local_update_norm_track, "LocalUpdateNorm", args.dataset, t, writer) # List[List[float]], length: com_round


    writer.close()
    torch.save(handler._model.state_dict(), os.path.join(path, "model.pth"))
