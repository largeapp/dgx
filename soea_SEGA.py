# -*- coding: utf-8 -*-
import geatpy as ea  # 导入geatpy库
import numpy as np
import time 
from problems import Recnbr, Mutnbr 

class soea_SEGA_DGX(ea.SoeaAlgorithm):
    """
soea_SEGA_DGX : class - Strengthen Elitist GA Algorithm for DGX 
"""

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs): 
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing, trappedValue, maxTrappedCount, dirName)
        if population.ChromNum != 1: 
            raise RuntimeError('chromesome must be single structure')
        self.name = 'SEGA'
        self.selFunc = 'etour'                      # [dup,tour,ecs,etour]  
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=0.7)     
            self.mutOper = ea.Mutinv(Pm=0.5)        
        else:
            self.recOper = ea.Xovdp(XOVR=0.7)       
            if population.Encoding == 'BG': 
                self.mutOper = ea.Mutbin(Pm=None)  
            elif population.Encoding == 'RI':  
                self.mutOper = ea.Mutbga(Pm=1 / self.problem.Dim, MutShrink=0.5, Gradient=10)    
                self.recOper = ea.Recdis(RecOpt=0.7)  
                # self.recOper = ea.Recndx(XOVR=0.7)  
            else:
                raise RuntimeError('encoding must be one of ''BG'', ''RI'' ''P''.')
        self.distance_func = problem.distance_func 
        self.best_population = None 
        self.best_pop_objv = None 

    def stat(self, pop): 
        feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)  
        if len(feasible) > 0:
            feasiblePop = pop[feasible] 
            num_valid = np.sum(feasiblePop.ObjV[:,0] > 0.5)  
            mean_acc = np.mean(feasiblePop.ObjV[:,0]) 
            if num_valid == pop.sizes: 
                dist_mat = self.distance_func(feasiblePop) 
                diversity = np.mean(dist_mat, axis=1) 
                if self.best_population is not None: 
                    acc_diff = mean_acc - np.mean(self.best_population.ObjV[:,0])
                    div_diff = np.mean(diversity) - np.mean(self.best_pop_objv)
                if self.best_population is None or (acc_diff > -1e-2 and div_diff > 0):
                    self.best_population = feasiblePop 
                    self.best_pop_objv = diversity 
                    self.outFunc(self, pop, extra_msg="Better Generation") 
            bestIndi = feasiblePop[np.argmax(feasiblePop.FitnV)]   
            if self.BestIndi.sizes == 0:
                self.BestIndi = bestIndi  
            else:
                delta = (self.BestIndi.ObjV - bestIndi.ObjV) * self.problem.maxormins if \
                    self.problem.maxormins is not None else self.BestIndi.ObjV - bestIndi.ObjV
                
                self.trappedCount += 1 if np.abs(delta) < self.trappedValue else 0
                
                if delta > 0:
                    self.BestIndi = bestIndi
            
            self.trace['f_best'].append(bestIndi.ObjV[0][0])
            self.trace['f_avg'].append(np.mean(feasiblePop.ObjV))
            if self.logTras != 0 and self.currentGen % self.logTras == 0:
                self.logging(feasiblePop)  
                if self.verbose:
                    self.display()        
            self.draw(self.BestIndi)   
    
    def reinsertion(self, population, offspring, NUM): 
        population = population + offspring             
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins) 
        sorted_index = np.flip(np.argsort(population.FitnV, axis=0))  
        sorted_population = population[sorted_index] 
        distance_matrix = self.distance_func(sorted_population, ) 
        
        chooseIdx = [0]      
        candidates, duplicates = [], []  
        hist_distance = [0]       
        hist_len = NUM//3         
        
        for idx in range(1, len(sorted_population)): 
            distances = distance_matrix[idx, chooseIdx]    
            min_index = np.argmin(distances) 
            if distances[min_index] < 1e-4: 
                duplicates.append(idx) 
            
            if distances[min_index] > np.min(hist_distance): 
                if len(hist_distance) < hist_len: 
                    hist_distance.append(distances[min_index]) 
                else: 
                    hist_distance[np.argmin(hist_distance)] = distances[min_index] 
                if len(chooseIdx) >= NUM:      
                    if abs(sorted_population.ObjV[chooseIdx[min_index]][0] - sorted_population.ObjV[idx][0]) < 0.05:
                        chooseIdx.pop(min_index) 
                        chooseIdx.append(idx) 
                else: 
                    chooseIdx.append(idx) 
        remain = set(range(len(sorted_population))) - set(chooseIdx) - set(candidates) - set(duplicates)
        candidates.extend(list(remain) + duplicates)
        while len(chooseIdx) < NUM:
            chooseIdx.append(candidates[0])
            candidates.pop(0) 
        chooseFlag = np.zeros(population.sizes, dtype=bool, ) 
        chooseFlag[chooseIdx] = True 
        return sorted_population[chooseFlag]

    def run(self, prophetPop=None):         # prophet Population
        # ==========================init===========================
        population = self.population
        NIND = population.sizes
        self.initialization()         
        # ===========================prepare evolution============================
        population.initChrom(NIND)        
        
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  
        self.call_aimFunc(population)      
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  
        # ===========================start evolution============================
        while not self.terminated(population):
            # select
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            
            offspring.Chrom = self.recOper.do(offspring.Chrom)  
            offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  
            self.call_aimFunc(offspring)  
            
            # population = population[ea.selecting('dup', population.FitnV, NIND)]  
            population = self.reinsertion(population, offspring, NIND)   
        return self.finishing(population)  

    def finishing(self, pop):
        feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)  
        if len(feasible) > 0:
            feasiblePop = pop[feasible]
            if self.logTras != 0 and (len(self.log['gen']) == 0 or self.log['gen'][-1] != self.currentGen): 
                self.logging(feasiblePop)
                if self.verbose:
                    self.display()
        self.passTime += time.time() - self.timeSlot  
        self.draw(pop, EndFlag=True)  
        if self.plotter:
            self.plotter.show()
        
        return [self.best_population, pop]

class soea_psy_SEGA_DGX(ea.SoeaAlgorithm): 
    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs):
        # 先调用父类构造方法
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing, trappedValue, maxTrappedCount, dirName)
        if population.ChromNum == 1:
            raise RuntimeError('chromesome must be psy')
        self.name = 'psy-SEGA'
        self.selFunc = 'etour'  # [dup,tour,ecs,etour] 
        self.cfg = kwargs['cfg'] if 'cfg' in kwargs else None 
        Px, Pm = 0.8, 0.5      
        Pnode, Pnbr = None, None 
        if self.cfg is not None: 
            Px, Pm = self.cfg.xov_prob, self.cfg.mut_prob 
            Pnode, Pnbr = self.cfg.Pnode, self.cfg.Pnbr 
        
        self.recOper = Recnbr(Px, Pnode, Pnbr, problem, population, Half_N=population.sizes)
        self.mutOper = Mutnbr(Pm, Pnode, Pnbr, problem, population) 
        
        self.distance_func = problem.distance_func 
        self.obj_func = problem.objFunc
        self.fit_func = problem.fitFunc 
        self.best_population = None 
        self.best_pop_objv = None 

    def stat(self, pop): 
        
        feasible = np.all(pop.CV <= 0, 1) if pop.CV is not None else np.arange(pop.sizes) 
        if np.any(feasible): 
            feasiblePop = pop[feasible] 
            pop_objv = self.obj_func(feasiblePop, pop.sizes) 
            if self.best_population is None or pop_objv > self.best_pop_objv: 
                self.best_population = pop      
                self.best_pop_objv = pop_objv
                self.outFunc(self, pop, extra_msg="Better Generation") 
            bestIndi = feasiblePop[np.argmax(feasiblePop.FitnV)]  
            if self.BestIndi.sizes == 0:
                self.BestIndi = bestIndi  # init global best individual
            else:
                delta = (self.BestIndi.ObjV[:,0] - bestIndi.ObjV[:,0]) * self.problem.maxormins if \
                    self.problem.maxormins is not None else self.BestIndi.ObjV[:,0] - bestIndi.ObjV[:,0]
                
                self.trappedCount += 1 if np.abs(delta) < self.trappedValue else 0
                
                if delta > 0:
                    self.BestIndi = bestIndi
            
            self.trace['f_best'].append(bestIndi.ObjV[0][0])
            self.trace['f_avg'].append(np.mean(feasiblePop.ObjV[:,0]))
            if self.logTras != 0 and self.currentGen % self.logTras == 0:
                self.logging(feasiblePop)  
                if self.verbose:
                    self.display()       
            self.draw(self.BestIndi)   
    
    def reinsertion(self, population, offspring, NUM): 
        population = population + offspring         
        population.FitnV = ea.scaling(self.fit_func(population), population.CV, self.problem.maxormins) 
        sorted_index = np.flip(np.argsort(population.FitnV, axis=0))  
        sorted_population = population[sorted_index] 
        distance_matrix = self.distance_func(sorted_population, ) 
        
        chooseIdx = [0]      
        candidates, duplicates = [], []  
        hist_distance = [0]        
        hist_len = NUM//3               # 0 < hist_len <= NUM 
        
        for idx in range(1, len(sorted_population)): 
            distances = distance_matrix[idx, chooseIdx]    
            min_index = np.argmin(distances) 
            if distances[min_index] < 1e-4: 
                duplicates.append(idx) 
            
            if distances[min_index] > np.min(hist_distance): 
                if len(hist_distance) < hist_len: 
                    hist_distance.append(distances[min_index]) 
                else: 
                    hist_distance[np.argmin(hist_distance)] = distances[min_index] 
                if len(chooseIdx) >= NUM:    
                    if abs(sorted_population.ObjV[chooseIdx[min_index]][0] - sorted_population.ObjV[idx][0]) < 0.05:
                        chooseIdx.pop(min_index) 
                        chooseIdx.append(idx) 
                else: 
                    chooseIdx.append(idx) 
        remain = set(range(len(sorted_population))) - set(chooseIdx) - set(candidates) - set(duplicates)
        candidates.extend(list(remain) + duplicates)
        while len(chooseIdx) < NUM:
            chooseIdx.append(candidates[0])
            candidates.pop(0) 
        chooseFlag = np.zeros(population.sizes, dtype=bool, ) 
        chooseFlag[chooseIdx] = True 
        return sorted_population[chooseFlag]

    def run(self, prophetPop=None):         # prophet Population
        # =====================================================
        population = self.population
        NIND = population.sizes
        with_niche = self.cfg.niche if self.cfg is not None else 'no'
        self.initialization()             
        # =======================================================
        population.initChrom(NIND)      
        if prophetPop is not None:          
            population = (prophetPop + population)[:NIND]  
        self.call_aimFunc(population)                     
        population.FitnV = ea.scaling(self.fit_func(population), population.CV, self.problem.maxormins) 
        # =======================================================
        while not self.terminated(population):
            offspring = population[ea.selecting(self.selFunc, population.FitnV, NIND)]
            offspring.Chroms = self.recOper.do(offspring.Chroms)    
            offspring.Chroms = self.mutOper.do(offspring.Chroms, offspring.Encodings, offspring.Fields) 
            
            self.call_aimFunc(offspring)  
            if with_niche == "reinsertion": 
                population = self.reinsertion(population, offspring, NIND)     
            elif with_niche == "crowding": 
                population = population + offspring 
                population = population[ea.selecting('dup', population.FitnV, NIND)] 
            else:               # "no"
                population = offspring 
        return self.finishing(population)  

    def finishing(self, pop):
        feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)  
        if len(feasible) > 0:
            feasiblePop = pop[feasible]
            if self.logTras != 0 and (len(self.log['gen']) == 0 or self.log['gen'][-1] != self.currentGen):  
                self.logging(feasiblePop)
                if self.verbose:
                    self.display()
        self.passTime += time.time() - self.timeSlot 
        self.draw(pop, EndFlag=True)  
        if self.plotter:
            self.plotter.show()
        
        return [self.best_population, pop]