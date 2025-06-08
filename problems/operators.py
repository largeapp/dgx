# -*- coding: utf-8 -*-
import os 
import geatpy as ea  
import numpy as np
import torch
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data 
from config import Map 
from scipy.sparse import coo_matrix 
from typing import List 

class Recnbr(ea.Recombination): 
    """
    Recnbr - class  : 
            Px      : 
            Pnode   : 
            Half_N  :  
            GeneID  : 
            Parallel: 
    """
    def __init__(self, XOVR=None, Pnode=None, Pnbr=None, problem=None, population=None, Half_N=False, GeneID=None, Parallel=False):
        self.Pnode = Pnode 
        self.Pnbr = Pnbr 
        self.Half_N = Half_N  
        self.GeneId = GeneID 
        self.Parallel = Parallel 
        self.problem_name = problem.name 
        self.num_node = problem.num_nodes
        Px = XOVR if XOVR is not None else 1.0/(self.num_node) 
        if self.problem_name in ["MotifExplain", "MoleculeCustomExplain", "BrainNetworkExplain"]: 
            self.Px = np.asarray([Px] * problem.triu_len) 
            self.Pnode = Pnode if Pnode is not None else np.asarray([1.0/self.num_node] * self.num_node) 
            self.Pnbr = Pnbr if Pnbr is not None else np.asarray([1.0/self.num_node] * self.num_node) 
        elif self.problem_name == "BrainNetworkExplain_FC": 
            self.Px = Px  
            self.recOpers = [ea.Recdis(RecOpt=self.Px, Half_N=self.Half_N)] 
        else: 
            self.ChromNum = population.ChromNum
            self.recOpers = [] 
            for i in range(self.ChromNum): 
                # dim = population.Linds[i] 
                # Px = np.array([1.0] * dim) / dim 
                if population.Encodings[i] == 'P':
                    recOper = ea.Xovpmx(XOVR=Px, Half_N=self.Half_N)  
                elif population.Encodings[i] == 'BG':
                    recOper = ea.Xovdp(XOVR=Px, Half_N=self.Half_N)   
                elif population.Encodings[i] == 'RI':
                    recOper = ea.Recdis(RecOpt=Px, Half_N=self.Half_N) 
                    # recOper = ea.Recndx(XOVR=Px, Half_N=self.Half_N)  
                else:
                    raise RuntimeError('编码方式必须为''BG''、''RI''或''P''. ') 
                self.recOpers.append(recOper) 

    def do(self, OldChrom, *args): 
        if self.problem_name in ["MotifExplain", "MoleculeCustomExplain", "BrainNetworkExplain"]: 
            return self.recnbr(OldChrom, args) 
        elif self.problem_name == "BrainNetworkExplain_FC": 
            return [self.recOpers[0].do(OldChrom[0]), OldChrom[1]] 
        else: 
            # if self.ChromNum == 1: 
            #     return self.recOpers[0].do(Encoding[0], OldChrom[0], FieldDR)
            newChrom = []                
            for i in range(self.ChromNum):
                newChrom.append(self.recOpers[i].do(OldChrom[i], )) 
            return newChrom 

    def recnbr(self, OldChrom, *args): 
        def xovnbr_1(pa, pb): 
            o1_adjv, o2_adjv = OldChrom[pa], OldChrom[pb]
            select_node = np.random.choice(self.num_node, 1, True, p=self.Pnode/self.Pnode.sum()) 
            pos = self.locate(self.num_node, [select_node]) 
            tmp_adjv = o1_adjv[pos] 
            o1_adjv[pos] = o2_adjv[pos] 
            o2_adjv[pos] = tmp_adjv 
            return o1_adjv, o2_adjv 
        def xovnbr_2(pa, pb): 
            o1_adjv, o2_adjv = OldChrom[0][pa], OldChrom[0][pb]
            o1_feat, o2_feat = OldChrom[1][pa], OldChrom[1][pb] 
            if self.Pnbr.sum() > 0: 
                select_node = np.random.choice(self.num_node, 1, True, p=self.Pnbr/self.Pnbr.sum()) 
                if self.Pnode[select_node] > 0: 
                    tmp_feat = o1_feat[select_node] 
                    o1_feat[select_node] = o2_feat[select_node] 
                    o2_feat[select_node] = tmp_feat 
                pos = self.locate(self.num_node, [select_node]) 
                tmp_adjv = o1_adjv[pos] 
                o1_adjv[pos] = o2_adjv[pos] 
                o2_adjv[pos] = tmp_adjv 
            return [o1_adjv, o1_feat], [o2_adjv, o2_feat] 
        
        if isinstance(OldChrom, List) and len(OldChrom) == 2: 
            newChrom0, newChrom1 = [], []  
            pop_size= OldChrom[0].shape[0] 
            if self.Half_N > 1: 
                offsp_size = min(self.Half_N, pop_size) 
                for i in range(offsp_size): 
                    pid = np.random.choice(pop_size, 2, False)
                    o1, _ = xovnbr_2(min(pid), max(pid)) 
                    newChrom0.append(o1[0])
                    newChrom1.append(o1[1])
            else:       # Half_N: True or False (1 or 0) 
                offsp_size = pop_size // 2 
                p1, p2 = np.arange(0, offsp_size), np.arange(offsp_size, 2*offsp_size) 
                for i in range(offsp_size): 
                    o1, o2 = xovnbr_2(p1[i], p2[i])
                    if self.Half_N == True: 
                        newChrom0.append(o1[0]) 
                        newChrom1.append(o1[1]) 
                    else: 
                        newChrom0 += [o1[0], o2[0]]
                        newChrom1 += [o1[1], o2[1]]
            # print(f"{self.Half_N}, {pop_size}, {offsp_size}, {len(newChrom0)} ") 
            return [np.vstack(newChrom0), np.vstack(newChrom1)] 
        else: 
            newChrom = [] 
            pop_size = OldChrom.shape[0]
            if self.Half_N > 1: 
                offsp_size = min(self.Half_N, pop_size) 
                for i in range(offsp_size): 
                    pid = np.random.choice(pop_size, 2, False) 
                    newChrom.append(xovnbr_1(min(pid), max(pid))[0] )
            else:       # Half_N: True or False (1 or 0)
                offsp_size = pop_size//2 
                p1, p2 = np.arange(0, offsp_size), np.arange(offsp_size, 2*offsp_size) 
                for i in range(offsp_size): 
                    o1, o2 = xovnbr_1(p1[i], p2[i])
                    if self.Half_N == True: 
                        newChrom0.append(o1) 
                    else: 
                        newChrom0 += [o1, o2]
            return newChrom 
        # return recdis(OldChrom, self.RecOpt, self.Half_N, self.GeneID, self.Parallel)
    
    def locate(self, N, selected): 
        # N: num of nodes; selected: list of selected nodes (0,N-1) [n1,...]
        triu_index = np.triu_indices(N, k=1) 
        adj = np.zeros((N,N), dtype=bool) 
        adj[selected, :] = True 
        adj[:, selected] = True 
        return adj[triu_index[0], triu_index[1]] 

    def getHelp(self): 
        # help(mutuni)
        pass

class Mutnbr(ea.Mutation): 
    """
    Mutnbr - class : 
            Pm     : 
            Pnode  : 
            FixType: 
            Parallel: 
    """
    def __init__(self, Pm=None, Pnode=None, Pnbr=None, problem=None, population=None, FixType=1, Parallel=False):
        self.Pnode = Pnode 
        self.FixType = FixType
        self.Parallel = Parallel 
        self.problem_name = problem.name 
        self.num_node = problem.num_nodes
        Pm = Pm if Pm is not None else 1.0/(self.num_node)
        if self.problem_name in ["MotifExplain", "BrainNetworkExplain"]: 
            self.Pm = np.asarray([Pm ]*(problem.triu_len-problem.csm_triu_len) + [0]*problem.csm_triu_len) 
            self.Pnode = Pnode if Pnode is not None else np.asarray([1.0/self.num_node] * self.num_node) 
            self.Pnbr = Pnbr if Pnbr is not None else np.asarray([1.0/self.num_node] * self.num_node) 
            # self.recOpers = [ea.Xovdp(XOVR=Px), ea.Recdis(RecOpt=Px)] 
            # self.mutOpers = [ea.Mutuni(Pm=Pm, Alpha=False, Middle=True), ea.Mutuni(Pm=Pm, Alpha=False, Middle=True)] 
            self.options = {"eps": -4.5} 
        elif self.problem_name == "MoleculeCustomExplain": 
            self.Pm = np.asarray([Pm ]*(problem.triu_len-problem.csm_triu_len) + [0]*problem.csm_triu_len) 
            self.Pnode = Pnode if Pnode is not None else np.asarray([1.0/self.num_node] * self.num_node) 
            self.Pnbr = Pnbr if Pnbr is not None else np.asarray([1.0/self.num_node] * self.num_node) 
            self.options = {"max_node_deg":problem.valency} 
        elif self.problem_name == "BrainNetworkExplain_FC": 
            # self.Pm = Pm if Pm is not None else 1.0/(self.num_node)
            ## Mutuni(Pm=Pm, Alpha=1, Middle=False), Mutgau(Pm, Sigma3=0.02), 
            # self.mutOpers = [ea.Mutbga(Pm=Pm, MutShrink=0.5, Gradient=4)] 
            self.mutOpers = [ea.Mutmove(Pm=Pm, MoveLen=116, Pr=0.5)] 
        else: 
            self.ChromNum = population.ChromNum
            self.mutOpers = [] 
            for i in range(self.ChromNum):
                # dim = population.Linds[i] if self.ChromNum>1 else population.Lind 
                # Pm = np.array([Pm] * dim) 
                if population.Encodings[i] == 'P':
                    mutOper = ea.Mutinv(Pm=Pm)   
                else:
                    if population.Encodings[i] == 'BG':
                        mutOper = ea.Mutbin(Pm=None) 
                    elif population.Encodings[i] == 'RI':
                        mutOper = ea.Mutbga(Pm=Pm, MutShrink=0.5, Gradient=10)    # breeder (0.5, 20)
                    else:
                        raise RuntimeError('encoding must be: ''BG''、''RI''或''P''. ') 
                self.mutOpers.append(mutOper) 

    def do(self, OldChrom, Encoding, FieldDR, *args): 
        if self.problem_name in ["MotifExplain", "MoleculeCustomExplain", "BrainNetworkExplain"]: 
            return self.mutnbr(OldChrom, Encoding, FieldDR, **self.options) 
        elif self.problem_name == "BrainNetworkExplain_FC": 
            return [self.mutOpers[0].do(Encoding[0], OldChrom[0], FieldDR[0]), OldChrom[1]] 
        else: 
            # print(f"type: {type(FieldDR)}")
            # if self.ChromNum == 1: 
            #     return self.mutOpers[0].do(Encoding[i], OldChrom[i], FieldDR[0])
            newChrom = [] 
            for i in range(self.ChromNum):
                newChrom.append(self.mutOpers[i].do(Encoding[i], OldChrom[i], FieldDR[i])) 
            return newChrom 
        # return mutuni(Encoding, OldChrom, FieldDR, self.Pm, self.Alpha, self.Middle, self.FixType, self.Parallel)
    
    def mutnbr(self, OldChrom, Encoding, FieldDR, **args): 
        def excute(triu, node, max_deg=None): 
            pos = self.locate(self.num_node, [node])  
            cur_deg = triu[pos].sum()    
            fix_deg = triu[(self.Pm==0) & pos].sum()  
            if max_deg is not None:     
                min_deg = max(fix_deg, 1) 
                pdf = np.array([(i+2)**(-1.5) for i in range(max_deg[node]-min_deg+1)], dtype=np.float32) 
                node_deg = np.random.choice(a=np.arange(min_deg, max_deg[node]+1), p=pdf/pdf.sum())  
            else: 
                pdf = np.array([(i+2)**args['eps'] for i in range(self.num_node-1)], dtype=np.float32) 
                node_deg = np.random.choice(a=np.arange(1, self.num_node), p=pdf/pdf.sum())
            node_deg = max(node_deg-fix_deg, 0) 
            mod_pos = (self.Pm > 0) & pos  
            if max_deg is not None: 
                triu[mod_pos] = 0 
                adj = np.zeros((self.num_node, self.num_node), dtype=np.int32) 
                adj[np.triu_indices(self.num_node, 1)] = triu 
                adj += adj.T 
                degree = np.delete(max_deg-adj.sum(axis=1), [node]) 
                mod_pos[pos] = mod_pos[pos] & (degree > 0) 
            p_edge = self.Pm[mod_pos] 
            if p_edge.sum() > 0: 
                select_edge = np.random.choice(np.where(mod_pos)[0], min(node_deg, np.sum(p_edge>0)), False, p=p_edge/p_edge.sum()) 
                triu[select_edge] = 1  
            return triu 
        
        if isinstance(OldChrom, List) and len(OldChrom) == 2: 
            adjv, nodefeat = OldChrom[0], OldChrom[1] 
            pop_size = adjv.shape[0] 
            for i in range(pop_size): 
                if self.Pnode.sum() > 0:    
                    select_node = np.random.choice(self.num_node, p=self.Pnode/self.Pnode.sum()) 
                    lb, ub = FieldDR[1][:2, select_node] 
                    if FieldDR[1][2][select_node] == 0:  
                        nodefeat[i][select_node] = np.random.uniform(lb, ub, 1)      # [lb,ub) 
                    else: 
                        nodefeat[i][select_node] = np.random.random_integers(int(lb), int(ub), 1)  
                        max_deg = [args['max_node_deg'][n] for n in nodefeat[i]] if 'max_node_deg' in args else None 
                        adjv[i] = excute(adjv[i], select_node, max_deg) 
                if self.Pnbr.sum() > 0:     
                    select_node = np.random.choice(self.num_node, p=self.Pnbr/self.Pnbr.sum()) 
                    max_deg = [args['max_node_deg'][n] for n in nodefeat[i]] if 'max_node_deg' in args else None 
                    adjv[i] = excute(adjv[i], select_node, max_deg) 
            return [adjv, nodefeat] 
        else: 
            adjv = OldChrom 
            pop_size = OldChrom.shape[0] 
            for i in range(pop_size): 
                if self.Pnbr.sum() > 0:
                    select_node = np.random.choice(self.num_node, p=self.Pnbr/self.Pnbr.sum()) 
                    adjv[i] = excute(adjv[i], select_node ) 
            return adjv 
    
    def locate(self, N:int, selected:List): 
        # N: num of nodes; selected: list of selected nodes (0,N-1) [n1,...]
        triu_index = np.triu_indices(N, k=1) 
        adj = np.zeros((N,N), dtype=bool) 
        adj[selected, :] = True 
        adj[:, selected] = True 
        return adj[triu_index[0], triu_index[1]] 

    def getHelp(self):  
        # help(mutuni)
        pass




