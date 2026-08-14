class StoppingController:
    def __init__(self,cfg): self.cfg=cfg['stopping']; self.best=-1.; self.wait=0
    def update(self,round_idx,accuracy,elapsed,max_rounds):
        mode=self.cfg['mode']
        if mode=='fixed_rounds': return (round_idx>=max_rounds,'fixed_rounds' if round_idx>=max_rounds else None)
        if mode=='target_accuracy':
            if accuracy is None: return False,None
            return (accuracy>=float(self.cfg['target_accuracy']),'target_accuracy' if accuracy>=float(self.cfg['target_accuracy']) else None)
        if mode=='runtime_budget': return (elapsed>=float(self.cfg['runtime_budget_sec']),'runtime_budget' if elapsed>=float(self.cfg['runtime_budget_sec']) else None)
        if mode=='convergence':
            if accuracy is None: return False,None
            tol=float(self.cfg['convergence_tolerance']); pat=int(self.cfg['patience'])
            if accuracy>self.best+tol: self.best=accuracy; self.wait=0
            else: self.wait+=1
            return (self.wait>=pat,'convergence' if self.wait>=pat else None)
        raise ValueError(mode)
    def state_dict(self): return {'best':self.best,'wait':self.wait}
    def load_state_dict(self,state): self.best=float(state.get('best',-1.)); self.wait=int(state.get('wait',0))
