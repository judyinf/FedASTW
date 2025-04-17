"""
This module defines the Aggregators class, which contains methods for aggregating model parameters.
"""

import torch

class Aggregators(object):
    """Define the algorithm of parameters aggregation"""

    @staticmethod
    def fedastw_aggregate(layerwise_params_list, weights=None, layer=5):
        """FedASTW aggregation algorithm.
        Args:
            layerwise_params_list (List[List[torch.Tensor]]): list of params for each layer, (round_clients,layer).
            weights (List[List[Optional(float,int)]]): list of weights for each layer, (round_clients,layer).
                If None, all weights are set to 1.
            layer (int): number of layers, default is 5.

        Returns:
            torch.Tensor
        """
        # Check if the input layerwise_params_list is correctly formatted
        if not isinstance(layerwise_params_list, list) or not all(isinstance(i, list) for i in layerwise_params_list):
            raise ValueError("layerwise_params_list must be a list of lists.")
        if not all(isinstance(i, torch.Tensor) for sublist in layerwise_params_list for i in sublist):
            raise ValueError("All elements of layerwise_params_list must be torch.Tensor.")
        if not all(len(sublist) == layer for sublist in layerwise_params_list):
            raise ValueError("Each sublist in layerwise_params_list must have the same length equal to the number of layers.")
        
        if not weights:
            weights = [[1]*layer for _ in range(len(layerwise_params_list))] # default weights

        # Check if the input weights is correctly formatted
        if not isinstance(weights, list) or not all(isinstance(i, list) for i in weights):
            raise ValueError("weights must be a list of lists.")
        if not len(weights) == len(layerwise_params_list):
            raise ValueError("The length of weights and layerwise_params_list must be the same.")
        if not all(isinstance(i, (float, int)) for sublist in weights for i in sublist):
            raise ValueError("All elements of weights must be float or int.")
        if not all(len(sublist) == layer for sublist in weights):
            raise ValueError("Each sublist in weights must have the same length equal to the number of layers.")
        
        aggregated_params_list = []
        
        for i in range(layer):
            layer_weight = [weight[i] for weight in weights]
            # check if the weights are non-negative
            if any(w < 0 for w in layer_weight):
                raise ValueError(f"Layer {i} weights must be non-negative.")
            # normalize the weights
            layer_weight = torch.tensor(layer_weight)/sum(layer_weight)

            layer_params = [params[i] for params in layerwise_params_list]
            
            layer_aggregated_params = torch.sum(torch.stack(layer_params, dim=-1) * layer_weight, dim=-1)
            aggregated_params_list.append(layer_aggregated_params)
        
        serialized_parameters = torch.cat(aggregated_params_list, dim=0)
        
        # Check if the serialized_parameters is a tensor
        if not isinstance(serialized_parameters, torch.Tensor):
            raise ValueError("The serialized_parameters must be a torch.Tensor.")
        # Check if the serialized_parameters is not empty
        if serialized_parameters.numel() == 0:
            raise ValueError("The serialized_parameters must not be empty.")
        
        return aggregated_params_list
    
    
    @staticmethod
    def fedavg_aggregate(serialized_params_list, weights=None):
        """FedAvg aggregator

        Paper: http://proceedings.mlr.press/v54/mcmahan17a.html

        Args:
            serialized_params_list (list[torch.Tensor])): Merge all tensors following FedAvg.
            weights (list, numpy.array or torch.Tensor, optional): Weights for each params, the length of weights need to be same as length of ``serialized_params_list``

        Returns:
            torch.Tensor
        """
        if weights is None:
            weights = torch.ones(len(serialized_params_list))

        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        weights = weights / torch.sum(weights)
        assert torch.all(weights >= 0), "weights should be non-negative values"
        serialized_parameters = torch.sum(
            torch.stack(serialized_params_list, dim=-1) * weights, dim=-1)

        return serialized_parameters

# test the Aggregators class
# if __name__ == "__main__":
#     # Example usage
#     layerwise_params_list = [
#         [torch.tensor([1.0, 2.0]), torch.tensor([2.0]), torch.tensor([3.0,4.0,5.0]),torch.tensor([3.0])],
#         [torch.tensor([4.0, 5.0]), torch.tensor([6.0]), torch.tensor([8.0, 9.0,0.0]),torch.tensor([6.0])]
#     ] # 2 clients, 4 layers
#     # Example weights
#     weights = [
#         [0.1, 0.2, 0.3, 0.4],
#         [0.5, 0.6, 0.7, 0.8]
#     ] # 2 clients, 4 layers
#     # Call the aggregation method
#     aggregated_params = Aggregators.fedastw_aggregate(layerwise_params_list, weights, layer=4)
#     print(aggregated_params)
#     # Example with default weights
#     aggregated_params_default = Aggregators.fedastw_aggregate(layerwise_params_list, layer=4)
#     print(aggregated_params_default)

    