# -*- coding: utf-8 -*-
import os, sys, csv 
from sacred import Experiment 
from sacred.utils import apply_backspaces_and_linefeeds 
from sacred import SETTINGS 
from sacred.observers import FileStorageObserver

from soea_SEGA import soea_SEGA_DGX, soea_psy_SEGA_DGX 
from problems import DGXSoeaPsyProblem, ABIDEProblem, MoleculePsyProblem  
from models import GCN, GIN, DiffPool, GAGCN, RESTDataset  
from config import Map, get_logger 
from util.plot import plot_graphs, get_hex_colors

import geatpy as ea 
import numpy as np 
import pandas as pd 
import torch 
import random, warnings
warnings.filterwarnings("ignore")


ex = Experiment() 
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.observers.append(FileStorageObserver(basedir="./result/sacred_logs", copy_sources=False)) 
ex.logger = get_logger() 
SETTINGS.CONFIG.READ_ONLY_CONFIG = False 


@ex.config
def DGX_config():
    """ The core configuration. """ 
    ex_name = "n5_052901"       # str, 
    ds_name = "motif2"          # str, 
    target_class = 0            # int, 
    niche = "reinsertion"       # str, [reinsertion, crowding, no] 
    candidate_types = "0,1,2,3,4" 
    max_num_nodes = 5
    customized_nodes = None     #  [num_custom, ] (0 < num_custom <= max_num_nodes), default:None 
    customized_struct = None    #  adjV: [num_custom*(num_custom-1)//2, ], default:None
    population_num = 10         # int, 
    _lambda = 0.8               # float, 
    mut_prob = 0.5              # float, 
    xov_prob = 0.5              # float, 
    epochs = 1000               # int, 
    logTrans = 10               # int, 
    repetition = 5              # int, 
    current_seed = None         # int, 
    Pnode = None 
    Pnbr = None 

    sourcedir = r"../../.tmp"
    targetdir = r"./result"
    res_dir = os.path.join(targetdir, ds_name, f"{ex_name}_x{target_class}")
    model_name = "gin_motif2_23052604" 
    best_fold = "Consensus"         # str, "Consensus", xxx
    model_dir = os.path.join(targetdir, 'checkpoints', model_name) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    hidden_dim = 32 
    num_layers = 3 
    concat_last = 1 
    num_node_features = None     
    num_classes = None          
    fc_threshold = None 
    sparse = None 
    need_prophet = False 
    sacred_log_dir = None 
    description = None 

@ex.named_config
def motif2():                       # run: python main_multi.py with motif2
    ds_name = "motif2" 
    ex_name = "n8m1kc_24101402_x0"
    model_name = "gin_motif2_24101301" 
    best_fold = "Consensus"         # Consensus, 
    niche = "reinsertion"           # [reinsertion, crowding, no]
    hidden_dim = 32 
    num_layers = 3 
    target_class = 0                # int, 
    epochs = 2000                   # int, 
    logTrans = 20                   # int, 
    repetition = 10                 # int, 
    population_num = 6 
    _lambda = 1.2
    max_num_nodes = 8
    candidate_types = "0,1,2,3,4" 
    customized_nodes = None        
    customized_struct = None    
    need_prophet = True 
    description = "use 10-fold gnn models, average probs; adjust single/multiple-nodes pattern to ±10 while eval diversity"

@ex.named_config
def mutagenicity():                 # run: python main_multi.py with mutagenicity
    ds_name = "Mutagenicity"
    ex_name = "val_n4m2kc_24101302" 
    model_name = "gin_Mutagenicity_23041402"
    best_fold = "Consensus"         # '_kf5_best' 
    niche = "reinsertion"           # [reinsertion, crowding, no]
    hidden_dim = 64 
    num_layers = 4 
    ## 'C.4', 'O.2', 'Cl.1', 'H.1', 'N.3', 'F.1', 'Br.1, 'S.6'
    candidate_types = "0,1,2,3,4,7" 
    max_num_nodes = 4 
    population_num = 10 
    repetition = 5 
    epochs = 1000 
    logTrans = 20                   # int, 
    target_class = 0 
    _lambda = 0.8 
    # [num_custom, ] (0 < num_custom <= max_num_nodes), default:None 
    customized_nodes = None   
    # adjV: [num_custom*(num_custom-1)//2, ], default:None
    customized_struct = None
    # customized_struct = "1,0,0,0,2,2,0,0,0,1,0,0,2,0,1"     # Aromatic
    # customized_struct = "1,0,0,0,0,1,0,1,0,1,0,1,0,0,0"     # Alkane
    mut_prob = 0.1 
    xov_prob = 0.8
    description = "with valency constraint for molecule patterns"

@ex.named_config
def abide1():                       # run: python main_multi.py with abide1 
    ds_name = "abide_aal" 
    ex_name = "n116m3kc_aal_062803" 
    model_name = "gcn_abideAAL_23062603"   #
    best_fold = 'Consensus'         # '_kf2_best' "Consensus"
    niche = "reinsertion"           # [reinsertion, crowding, no] 
    hidden_dim = 32 
    num_layers = 3 
    concat_last = 2 
    candidate_types = ",".join([str(i) for i in range(0,116)])
    max_num_nodes = 116 
    population_num = 20 
    repetition = 5 
    epochs = 1000 
    logTrans = 20 
    target_class = 0 
    _lambda = 0.1 
    customized_nodes = candidate_types
    customized_struct = None 
    mut_prob = 0.9                  # float, 
    xov_prob = 0.2                  # float, 
    fc_threshold = 0.5 
    sparse = 20 
    need_prophet = True 
    description = "find top edges over 116 ROIs, at least 10, r1=0.2, r2=0.2"    # "with constraints: connection=1; 7 Cerebrum + 3 Cerebellar"

@ex.capture
def load_dataset(_config, _log): 
    cfg = Map(_config) 
    if cfg.ds_name in [ 'Mutagenicity']:
        from torch_geometric.datasets import TUDataset
        rootdir = os.path.join(cfg.sourcedir, 'TUDataset')
        dataset = TUDataset(root=rootdir, name=cfg.ds_name)
    elif cfg.ds_name in ['motif2']: 
        from data_gen import SyntheticDataset 
        dataset = SyntheticDataset(cfg.sourcedir, cfg.ds_name, )
    elif cfg.ds_name in ['abide_aal']: 
        rootdir = os.path.join(cfg.sourcedir, 'nilearn_data')
        dataset = RESTDataset(rootdir, cfg.ds_name)
    # dataset = dataset.shuffle() 
    _config.update(num_node_features=dataset.num_node_features) 
    _config.update(num_classes=dataset.num_classes) 
    _log.info(f"Dataset {cfg.ds_name} loaded: {dataset}") 
    return dataset 

@ex.capture 
def load_model(_config, fold=None): 
    cfg = Map(_config) 
    fold = cfg.best_fold if fold is None else fold 
    # create model and load the trained model weights
    if cfg.model_name.startswith('gcn'):
        model = GCN(input_dim=cfg.num_node_features,
                    hidden_dim=cfg.hidden_dim,
                    num_layers=cfg.num_layers,
                    num_classes=cfg.num_classes,
                    dropout=0.2,
                    concat_last=cfg.concat_last,
                    readout='g_attn',
                    seed=cfg.seed)
    elif cfg.model_name.startswith('gagcn'): 
        model = GAGCN(input_dim=cfg.num_node_features, 
                    hidden_dim=cfg.hidden_dim, 
                    num_layers=cfg.num_layers, 
                    num_classes=cfg.num_classes, 
                    dropout=0.2,
                    readout='g_attn',
                    concat_last=cfg.concat_last,
                    seed=cfg.seed)
    elif cfg.model_name.startswith('gin'): 
        model = GIN(input_dim=cfg.num_node_features,
                    hidden_dim=cfg.hidden_dim,
                    num_layers=cfg.num_layers,
                    num_classes=cfg.num_classes,
                    dropout=0.2,
                    JK='sum',
                    readout='g_attn',
                    seed=cfg.seed)
    else: 
        raise ValueError(f"gnn model must be one of [gcn, gin], got {cfg.model_name}")
    model.load_state_dict(torch.load(os.path.join(
        cfg.model_dir, 'model', f'model{fold}.pth'), map_location=torch.device('cpu')))
    model.eval() 
    return model 

def eval_diversity(cfg, pyg_graphs): 
    
    from util.evaluate import SumDistance, SumDistanceNN, PeakRatio, PeakDistance
    from util.tools import pyg2gkl 
    # gkl_list = pyg2gkl(pyg_graphs) 
    
    SDNN, PR, PR005, PD = -1, -1, -1, -1 
    gkl_kernel = "subgraph"             # graph kernel: [ subgraph, graphlet, wl, ]
    SD = SumDistance(pyg_graphs, cfg.ds_name, kernel=gkl_kernel, verbose=False) 
    if cfg.ds_name in ["motif1", 'motif1_2']: 
        SDNN = SumDistanceNN(pyg_graphs, cfg.ds_name, kernel=gkl_kernel)
        PR, peak_sim = PeakRatio(pyg_graphs, cfg.ds_name, r=0.1, kernel=gkl_kernel, verbose=False)
        PR005, peak_sim2 = PeakRatio(pyg_graphs, cfg.ds_name, r=0.05, kernel=gkl_kernel, verbose=False) 
        PD = PeakDistance(pyg_graphs, cfg.ds_name, kernel=gkl_kernel)
    return SD, SDNN, PD, PR, PR005 

def generate_prophet(Encodings, Fields, NIND, max_nodes):   
    prophet = ea.PsyPopulation(Encodings, Fields, NIND) 
    prophet.initChrom(NIND) 
    adj = np.zeros_like(prophet.Chroms[0], dtype=int)   # [NIND, triu_len]
    # adj = prophet.Chroms[0] 
    # adj[adj > 0.5] = np.random.rand()/10 + 0.4
    for i in range(NIND): 
        # for abide1 
        # sz = np.random.randint(max_nodes, (max_nodes-1)*int(np.log2(max_nodes)))  
        # sz = np.random.randint(10, 50) 
        # adj[i, np.random.choice(adj.shape[1], sz)] = 1 
        # adj[i, np.random.choice(adj.shape[1], sz)] = np.random.randint(50, 100, sz) / 100
        # for motif2 
        adj[i, [0,7,13,18,21,22,25,27]] = 1      # [0,3,4,7,9]
    prophet.Chroms[0] = adj 
    return prophet 

@ex.capture
def run(_config, _log, problem, rnd):
    """======================================""" 
    new_seed = np.random.randint(100000000) 
    round_dir = os.path.join(_config['res_dir'], f"r{rnd}_seed{new_seed}") 
    os.makedirs(round_dir, exist_ok=True) 
    _config.update(current_seed=new_seed) 
    random.seed(new_seed)
    np.random.seed(new_seed)
    torch.manual_seed(new_seed)
    cfg = Map(_config) 
    _log.info(f"Start run_{rnd} with seed={new_seed}") 

    """=======================================================""" 
    # 1. Motif2 explain: 
    # 2. Mutagenicity explain: 
    Encodings = ['RI', 'RI' ] 
    triu_len = problem.triu_len 
    # print(f"problem: {problem.varTypes.shape}, {problem.ranges.shape}, {problem.borders.shape} ")
    Field1 = ea.crtfld(Encodings[0], problem.varTypes[:triu_len], problem.ranges[:,:triu_len], problem.borders[:,:triu_len])  
    Field2 = ea.crtfld(Encodings[1], problem.varTypes[triu_len:], problem.ranges[:,triu_len:], problem.borders[:,triu_len:]) 
    Fields = [Field1, Field2]  
    
    population = ea.PsyPopulation(Encodings, Fields, cfg.population_num) 
    # 3. ABIDE-I explain: 
    prophet = None 
    if cfg.need_prophet: 
        prophet = generate_prophet(Encodings, Fields, cfg.population_num, cfg.max_num_nodes) 

    """====================================================="""
    
    def outFunc(alg, pop, log=_log, extra_msg="Generation"): 
        cv = np.all(pop.CV<=0, 1).sum() if pop.CV is not None else pop.sizes 
        spr = f'{np.mean(pop.ObjV[:,2]):.4f}' if pop.ObjV.shape[1] > 2 else 'none'
        message = f'{extra_msg} {alg.currentGen}: acc={np.mean(pop.ObjV[:,0]):.4f} dis={np.mean(pop.ObjV[:,1]):.4f} sparse={spr} cv={cv}' 
        if alg.best_population is not None: 
            message += f'\t\t| best_acc={alg.best_population.ObjV[:,0].mean():.4f} | best_objv={alg.best_pop_objv:.4f}'
        # message += f"\n\tcur_acc: {pop.ObjV[:,0]} fitv: {pop.FitnV[:,0]}" 
        log.info(message) 
    algorithm = soea_psy_SEGA_DGX(problem, population, outFunc=outFunc, cfg=cfg) 

    # algorithm.mutOper.Pm = cfg.mut_prob    
    # algorithm.recOper.XOVR = cfg.xov_prob  
    algorithm.MAXGEN = cfg.epochs         
    algorithm.logTras = cfg.logTrans     
    algorithm.verbose = False            
    algorithm.drawing = 0               
    algorithm.dirName = round_dir     

    """=====================================================""" 
    [BestPop, FinalPop] = algorithm.run(prophet)  
    population = BestPop if BestPop is not None else FinalPop 
    # BestPop.save(dirName=os.path.join(round_dir, 'best_ind')) 
    population.save(dirName=os.path.join(round_dir, 'final_pop')) 

    # visualize
    pyg_graphs = problem._construct_graph(population)
    total_label = [f"E{population.Chroms[0][i].sum()}_P{population.ObjV[i,0]:.4f}" for i in range(population.sizes)] 
    color_map = get_hex_colors(num_color=cfg.num_node_features)
    plot_graphs(pyg_graphs, round_dir, total_label, color_map, rows=2, cols=5, 
                preffix="DGX", rm_isolate=True, ds_name=cfg.ds_name)    
    torch.save({'data': pyg_graphs, 'label':total_label}, os.path.join(round_dir, 'x_result.pt'))

    # _log.info(f"final predict probs: {population.ObjV[:,0]}")
    VAL = np.mean(population.ObjV[:, 0]) 
    DIV = np.mean(population.ObjV[:, 1]) 
    SPR = np.mean(population.ObjV[:, 2]) 
    SD, SDNN, PD, PR, PR005 = eval_diversity(cfg, pyg_graphs) 
    _log.info(f"round_{rnd} - class_{cfg.target_class} metric:\n\
            VAL={VAL:.4f}\n\
            DIV={DIV:.4f}\n\
            SPR={SPR:.4f}\n\
            SD={SD:.4f}") 
    pd.DataFrame(data={"metric": ['VAL', 'DIV', 'SPR', 'SD', 'SDNN', 'PD', f'PR@0.1', 'PR@0.05'],
                        "value": [VAL, DIV, SPR, SD, SDNN, PD, PR, PR005]}
                        ).to_csv(os.path.join(round_dir, "metric.csv"), index=False) 
    with open(os.path.join(round_dir, 'args.csv'), 'w', newline='') as f:
        writer = csv.writer(f) 
        writer.writerows(_config.items())
    return [VAL, DIV, SPR, SD, SDNN, PD, PR, PR005]

@ ex.automain
def main(_config, _log):
    # Process config 
    os.makedirs(_config['res_dir'], exist_ok=True) 
    _config.update(sacred_log_dir=ex.observers[0].dir)
    _config.update(candidate_types=np.array([int(n) for n in _config['candidate_types'].split(',')])) 
    if _config['customized_nodes'] is not None: 
        _config.update(customized_nodes=np.array([int(n) for n in _config['customized_nodes'].split(',')])) 
        Pnode = np.ones(_config['max_num_nodes'], dtype=np.float32) 
        Pnode[-len(_config['customized_nodes']):] = 0. 
        _config.update(Pnode=Pnode) 
    if _config['customized_struct'] is not None: 
        _config.update(customized_struct=np.array([int(n) for n in _config['customized_struct'].split(',')]))
    _config.update(model_dir=os.path.join(_config['targetdir'], 'checkpoints', _config['model_name']) )

    _ = load_dataset()
    if _config['best_fold'] == "Consensus": 
        gnn_model = [] 
        for i in range(10): 
            gnn_model.append(load_model(fold=f"_kf{i}_best"))
    else: 
        gnn_model = load_model()

    """=================================================="""
    if _config['ds_name'] in ['abide_aal']: 
        DGX_problem = ABIDEProblem(gnn_model, _config) 
    elif _config['ds_name'] in ['Mutagenicity', ]: 
        DGX_problem = MoleculePsyProblem(gnn_model, _config) 
    else:          # motif2 
        DGX_problem = DGXSoeaPsyProblem(gnn_model, _config) 

    total_results = [] 
    for r in range(_config['repetition']): 
        total_results.append(run(problem=DGX_problem, rnd=r+1)) 
    
    # claculate mean&std result
    total_results = np.asarray(total_results)       # [R, M], 
    mean_res = np.mean(total_results, axis=0)
    std_res = np.std(total_results, axis=0)
    _log.info(f"Summary of {_config['repetition']} runs:\n\
          VAL={mean_res[0]:.4f}  ({std_res[0]:.4f})\n\
          DIV={mean_res[1]:.4f}  ({std_res[1]:.4f})\n\
          SPR={mean_res[2]:.4f}  ({std_res[2]:.4f})\n\
          SD={mean_res[3]:.4f}  ({std_res[3]:.4f})\n\
          SDNN={mean_res[4]:.4f}  ({std_res[4]:.4f})\n\
          PD={mean_res[5]:.4f}  ({std_res[5]:.4f})\n\
          PR={mean_res[6]:.4f}  ({std_res[6]:.4f})\n\
          PR@0.05={mean_res[7]:.4f}  ({std_res[7]:.4f})")
    res_df = pd.DataFrame(data={"Metric": ['VAL', 'DIV', 'SPR', 'SD', 'SDNN', 'PD', 'PR@0.1', 'PR@0.05']}) 
    for r in range(_config['repetition']): 
        res_df[f"r_{r+1}"] = total_results[r]
    res_df['mean'] = mean_res
    res_df['std'] = std_res 
    res_df.to_csv(os.path.join(_config['res_dir'], f"summary.csv"), index=False)


# if __name__ == '__main__': 
    # arguments 
    # main()
    # ex.run_commandline()









