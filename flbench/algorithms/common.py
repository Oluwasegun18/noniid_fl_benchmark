from flbench.training.hooks import LocalTrainingHooks
from flbench.training.local import train_local
from flbench.training.types import ClientUpdate

class StandardLocalAlgorithmMixin:
    def make_hooks(self,client_id,global_model,cfg): return LocalTrainingHooks()
    def make_context(self,client_id,global_model,cfg): return {'client_id':client_id}
    def client_update(self,client_id,global_model,loader,device,cfg):
        result=train_local(global_model,loader,device,cfg,self.make_hooks(client_id,global_model,cfg),self.make_context(client_id,global_model,cfg))
        return ClientUpdate(client_id,len(loader.dataset),result.state_dict,result.metrics,result.auxiliary)
