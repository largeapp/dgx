
"""
https://github.com/geatpy-dev/geatpy/blob/master/demo/soea_demo/soea_demo9 

optimization objection and constraints: 
max f = x * np.sin(10 * np.pi * x) + 2.0
s.t. -1 <= x <= 2
"""
import os 
import geatpy as ea 
import numpy as np
import torch
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data 
from config import Map 
from scipy.sparse import coo_matrix 
from typing import List 


class DGXSoeaProblem(ea.Problem): 
    def __init__(self, gnn_model, _config, **args): 
        cfg = Map(_config) 
        self.model = gnn_model 
        self.ds_name = cfg.ds_name 
        self.target_class = cfg.target_class
        self.nodes_type = cfg.nodes_type                
        self.node_dim = cfg.num_node_features           
        self.num_nodes = len(self.nodes_type)   
        self.node_feats = None 

        name = 'MotifExplain'
        Dim = self.num_nodes * (self.num_nodes - 1) // 2
        maxormins = np.array([-1] , dtype=int)      
        varTypes = np.ones(Dim, dtype=int)          
        lb = np.array([0]*Dim, dtype=int)
        ub = np.array([1]*Dim, dtype=int)
        lbin = np.ones(Dim, dtype=int)
        ubin = np.ones(Dim, dtype=int)
        super.__init__(self,
                        name,       
                        1,          
                        maxormins,  
                        Dim,        
                        varTypes,   
                        lb,         
                        ub,         
                        lbin,       
                        ubin        
        )
    
    def aimFunc(self, pop):
        Vars = pop.Chrom         
        
        func1 = np.zeros((pop.sizes, ), dtype=np.float32)

        pyg_graphs = self._construct_graph(pop)
        for i, data in enumerate(pyg_graphs):
            with torch.no_grad(): 
                logits = self.model(data)  
            func1[i] = logits.detach().softmax(dim=-1)[0,self.target_class] 
        upper_edges = self.num_nodes + np.round(np.log2(self.num_nodes)) 
        non_sparse = Vars.sum(axis=1) - upper_edges
        
        connects = np.array([self.count_connected_components(vec) 
                            for vec in Vars]) - 1 
        
        pop.CV = np.hstack([
            non_sparse[..., np.newaxis],         
            connects[..., np.newaxis],           
        ])
        pop.ObjV = np.hstack([func1[..., np.newaxis], ])
    
    def _get_node_feat(self, ):
        if self.node_feats is not None:
            return self.node_feats 
        
        self.node_feats = torch.zeros((self.num_nodes, self.node_dim), dtype=torch.float32)
        self.node_feats[range(self.num_nodes), self.nodes_type] = 1.
        return self.node_feats

    def _construct_graph(self, population, ):
        adjV = population.Chrom
        node_feat = self._get_node_feat()
        edge_weight = None
        data_list = []
        for adj_vec in adjV:
            if adj_vec.sum() == 0:         
                edge_index = torch.tensor([[], []], dtype=torch.long)
                data_list.append(Data(x=node_feat, edge_index=edge_index, edge_weight=edge_weight))
                continue
            k, edge_index = 0, []
            for i in range(0, self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    if adj_vec[k]:
                        edge_index.append([i, j])
                    k += 1 
            edge_index = torch.tensor(edge_index).T     
            edge_index = torch.hstack(
                (edge_index, edge_index[[1, 0]])).to(dtype=torch.long)
            data_list.append(Data(x=node_feat, edge_index=edge_index, edge_weight=edge_weight, ))
        return data_list 

    def get_grakel(self, kernel, ds_name, k=None): 
        import grakel 
        if k is None: 
            k = 5 if ds_name in ['motif1', 'motif1_2'] else 3 
        if kernel=="GraphletSampling": 
            gkl = grakel.GraphletSampling(normalize=True, random_state=42, k=k)
        elif kernel=="SubgraphMatching": 
            gkl = grakel.SubgraphMatching(normalize=True, k=k) 
        elif kernel=="RandomWalk": 
            gkl = grakel.RandomWalk(normalize=True, method_type="fast", kernel_type="geometric") 
        else: 
            gkl = grakel.WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=grakel.VertexHistogram)
        return gkl 

    def distance_func(self, population): 
        
        from grakel import Graph 
        gkl_list = [] 
        for data in self._construct_graph(population): 
            node_label = {i: lb for i,lb in enumerate(data.x.argmax(dim=-1)) }
            edges = [ (e[0], e[1]) for e in data.edge_index.T.numpy() ]
            edge_label = { e:(node_label[e[0]],node_label[e[1]]) for e in edges}
            gkl_list.append(Graph(edges, node_labels=node_label, edge_labels=edge_label))
        gkl = self.get_grakel('SubgraphMatching', self.ds_name) 
        distances = 1 - gkl.fit_transform(gkl_list) 
        return distances 

    def count_connected_components(self, adj_vec):  
        def find_root(roots, index):
            while roots[index] != index:    
                roots[index] = roots[roots[index]]  
                index = roots[index]
            return index
        N = int(np.ceil(np.sqrt(len(adj_vec) * 2)))
        roots = [i for i in range(N)]       
        count, k = N, 0                     
        for i in range(N):
            for j in range(i+1, N):         
                if adj_vec[k]:              
                    root_i = find_root(roots, i)
                    root_j = find_root(roots, j)
                    if root_i != root_j:    
                        roots[root_i] = root_j
                        count -= 1
                k += 1
        return count



class DGXSoeaPsyProblem(ea.Problem): 
    def __init__(self, gnn_model, _config, **args): 
        cfg = Map(_config) 
        self.model = gnn_model 
        self.ds_name = cfg.ds_name 
        self.target_class = cfg.target_class
        self._lambda = cfg._lambda if cfg.niche != 'no' else 0
        self.nodes_type = cfg.candidate_types           
        self.node_dim = cfg.num_node_features           
        self.num_nodes = cfg.max_num_nodes              
        self.triu_len = self.num_nodes * (self.num_nodes - 1) // 2   
        self.csm_nodes = cfg.customized_nodes           
        self.csm_struct = cfg.customized_struct         

        name = "MotifExplain_ori" if cfg.niche != 'no' else "MotifExplain_noniche"
        csm_node_num, csm_triu_len = 0, 0
        csm_node, csm_triu = [], [] 
        if self.csm_nodes is not None: 
            csm_node_num = len(self.csm_nodes) 
            csm_node = self.csm_nodes.tolist() 
        if self.csm_struct is not None: 
            csm_triu_len = self.csm_struct.shape[0]
            csm_triu = self.csm_struct.tolist() 
        self.csm_node_num = csm_node_num
        self.csm_triu_len = csm_triu_len 

        Dim = self.triu_len + self.num_nodes                
        varTypes = np.ones(Dim, dtype=int)                  
        maxormins = np.array([-1] , dtype=int)              
        lb = [0]*(self.triu_len-csm_triu_len) + csm_triu + [0]*(self.num_nodes-csm_node_num) + csm_node 
        ub = [1]*(self.triu_len-csm_triu_len) + csm_triu + [len(self.nodes_type)-1]*(self.num_nodes-csm_node_num) + csm_node
        lbin = [1] * Dim 
        ubin = [1] * Dim 
        
        ea.Problem.__init__(self,
                        name,       
                        3,          
                        maxormins,  
                        Dim,        
                        varTypes,   
                        lb,         
                        ub,         
                        lbin,       
                        ubin        
        )

    def validity(self, pop): 
        pyg_graphs = self._construct_graph(pop)
        valid = []                  
        for data in pyg_graphs:
            with torch.no_grad(): 
                if isinstance(self.model, List): 
                    logits = [self.model[i](data).detach().softmax(dim=-1).numpy() for i in range(len(self.model))]
                    logits = np.asarray(logits).mean(axis=0)                    
                else: 
                    logits = self.model(data).detach().softmax(dim=-1).numpy()  
                    
                    n_edge = data.edge_index.size(1) // 2                       
                    types = data.x.argmax(-1)
                    if n_edge > 7 or (n_edge==7 and not all(types.eq(types[0]))): 
                        logits = np.array([[0., 1.]], dtype=np.float32) 
            valid.append(logits)
        return np.vstack(valid, dtype=np.float32)[:, self.target_class] 

    def sparsity(self, pop):
        adjV = pop.Chroms[0]
        sparse = 1 - np.power(np.sum(adjV > 0, axis=1)/adjV.shape[1], 1/2) 
        return sparse
    
    def aimFunc(self, pop):
        
        validity = self.validity(pop)[..., np.newaxis]                  
        dissim = self.distance_func(pop).mean(axis=1)[..., np.newaxis]  
        sparse = self.sparsity(pop)[..., np.newaxis]                    
        pop.ObjV = np.hstack([validity, dissim, sparse]) 

        adjV, ntypes = pop.Chroms[0], pop.Chroms[1]   
        connects = np.array([self.count_connected_components(vec) for vec in adjV]) - 1 
        pop.CV = np.hstack([
            connects[..., np.newaxis],           
        ])
    
    def fitFunc(self, pop): 
        if '_noniche' in self.name:     
            return pop.ObjV[:, [0]] 
        return pop.ObjV[:, [0]] + pop.ObjV[:, [1]]*self._lambda + pop.ObjV[:, [2]]
    
    def objFunc(self, pop, max_pop=None): 
        
        max_pop = pop.sizes if max_pop is None else max_pop 
        if '_noniche' in self.name:     
            return pop.ObjV[:, 0].sum() / max_pop 
        obj_value = pop.ObjV[:, 0] + pop.ObjV[:, 1]*self._lambda + pop.ObjV[:, 2]
        return obj_value.sum() / max_pop 

    def _get_node_feat(self, feat): 
        
        node_feat = torch.zeros((feat.shape[0], self.num_nodes, self.node_dim), dtype=torch.float32) 
        for i, x in enumerate(feat): 
            types = self.nodes_type[x] 
            node_feat[i, range(self.num_nodes), types] = 1.  
        return node_feat  

    def _construct_graph(self, population, ):
        adjV = population.Chroms[0]                 
        adjM = np.zeros((adjV.shape[0], self.num_nodes, self.num_nodes), dtype=np.float32) 
        triu_idx = np.triu_indices(self.num_nodes, k=1) 
        for i in range(adjV.shape[0]):
            adjM[i][triu_idx] = adjV[i] 
            adjM[i] += adjM[i].T 
        node_feat = self._get_node_feat(feat=population.Chroms[1])  
        data_list = []
        for i, adj in enumerate(adjM): 
            adj = coo_matrix(adj) 
            if adj.data.shape[0] == 0:         
                edge_index = torch.tensor([[], []], dtype=torch.long) 
            else: 
                edge_index = torch.as_tensor(np.vstack((adj.row, adj.col)), dtype=torch.long) 
            data_list.append(Data(x=node_feat[i], edge_index=edge_index, 
                                  edge_attr=torch.as_tensor(adj.data, dtype=torch.float32).view(-1,1)))
        return data_list 

    def get_grakel(self, kernel, ds_name, k=None): 
        import grakel 
        if k is None: 
            k = 5 if ds_name in ['motif1', 'motif1_2', 'motif2'] else 3 
        if kernel=="GraphletSampling": 
            gkl = grakel.GraphletSampling(normalize=True, random_state=42, k=k)
        elif kernel=="SubgraphMatching": 
            gkl = grakel.SubgraphMatching(normalize=True, k=k) 
        elif kernel=="RandomWalk": 
            gkl = grakel.RandomWalk(normalize=True, method_type="fast", kernel_type="geometric") 
        else: 
            gkl = grakel.WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=grakel.VertexHistogram)
        return gkl 

    def distance_func(self, population): 
        
        from grakel import Graph 
        gkl_list = [] 
        flag = np.ones(population.sizes, dtype=bool).reshape(-1, 1) 
        for c, data in enumerate(self._construct_graph(population)): 
            if data.edge_index.size(1) == 0: 
                flag[c] = False 
                continue 
            if self.ds_name == 'motif2':  
                types = data.x.argmax(-1) 
                
                lb = 10 if all(types.eq(types[0])) else -10 
                node_label = {i: lb for i in range(data.x.size(0))}     
            else: 
                node_label = {i: lb for i,lb in enumerate(data.x.argmax(dim=-1).numpy()) } 
            edges = [ (e[0], e[1]) for e in data.edge_index.T.numpy() ]
            edge_label = { e:(node_label[e[0]],node_label[e[1]]) for e in edges} 
            gkl_list.append(Graph(edges, node_labels=node_label, edge_labels=edge_label))
        similarity = np.ones((population.sizes, population.sizes), dtype=np.float32) 
        if len(gkl_list) > 1: 
            gkl = self.get_grakel('SubgraphMatching', self.ds_name) 
            gkl_sim = gkl.fit_transform(gkl_list) 
            similarity[np.where(flag.T & flag)] = gkl_sim.reshape(-1) 
        return 1 - similarity  

    def count_connected_components(self, adj_vec):  
        def find_root(roots, index):
            while roots[index] != index:    
                roots[index] = roots[roots[index]]  
                index = roots[index]
            return index
        N = int(np.ceil(np.sqrt(len(adj_vec) * 2)))
        roots = [i for i in range(N)]       
        count, k = N, 0                     
        for i in range(N):
            for j in range(i+1, N):         
                if adj_vec[k]:              
                    root_i = find_root(roots, i)
                    root_j = find_root(roots, j)
                    if root_i != root_j:    
                        roots[root_i] = root_j
                        count -= 1
                k += 1
        return count


if __name__ == '__main__':
    problem = DGXSoeaProblem()
    Encoding = 'RI'  
    NINDs = [5, 10, 15, 20]  
    population = []  
    for i in range(len(NINDs)):
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) 
        population.append(ea.Population(Encoding, Field, NINDs[i]))  
    
    algorithm = ea.soea_multi_SEGA_templet(
        problem,
        population,
        MAXGEN=30,              
        logTras=2,              
        trappedValue=1e-6,      
        maxTrappedCount=5)      
    
    res = ea.optimize(algorithm,
                      verbose=True,
                      drawing=1,
                      outputMsg=True,
                      drawLog=False,
                      saveFlag=False)
    print(res)


