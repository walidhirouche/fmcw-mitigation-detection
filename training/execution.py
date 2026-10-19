"""Fixed-budget execution around the unchanged scientific training engine.

Only detached training diagnostics, plots and early-stop guards are removed.
The forward pass, losses, optimizer, scheduler and complete loaders are retained.
"""
import ast
import inspect
import textwrap
import types


def fit_source(engine):
    tree = ast.parse(textwrap.dedent(inspect.getsource(engine.Experiment.fit)))
    changes = dict(train_copy=0,train_match=0,train_summary=0,predict=0,evaluate=0,plots=0,stop=0,finished=0)
    class Transform(ast.NodeTransformer):
        def visit_Assign(self, node):
            target = ast.unparse(node.targets[0]) if len(node.targets)==1 else ''
            if target == 'probabilities':
                changes['train_copy']+=1; return None
            if target == 'train_metrics':
                if ast.unparse(node.value) != 'aggregate(train_rows)':
                    raise ValueError('Training diagnostic changed.')
                changes['train_summary']+=1
                node.value=ast.parse('dict(recall=None,fp_per_chirp=None)',mode='eval').body
            if target == 'finished':
                if ast.unparse(node.value) != 'epoch >= self.tc.epochs or self.bad_epochs >= self.tc.patience':
                    raise ValueError('Completion guard changed.')
                changes['finished']+=1
                node.value=ast.parse('epoch >= self.tc.epochs',mode='eval').body
            return self.generic_visit(node)
        def visit_For(self,node):
            if ast.unparse(node.target)=='(prob, meta)':
                if ast.unparse(node.iter)!="zip(probabilities, batch['meta'])":
                    raise ValueError('Training metric loop changed.')
                changes['train_match']+=1; return None
            return self.generic_visit(node)
        def visit_If(self,node):
            if ast.unparse(node.test)=='self.bad_epochs >= self.tc.patience':
                changes['stop']+=1; node.test=ast.Constant(False)
            return self.generic_visit(node)
        def visit_Expr(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and ast.unparse(node.value.func.value)=='plots':
                if node.value.func.attr not in ('learning_curves','examples'):
                    raise ValueError('Plot call changed.')
                changes['plots']+=1; return None
            return self.generic_visit(node)
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id in ('predict','evaluate_samples'):
                key='predict' if node.func.id=='predict' else 'evaluate'
                changes[key]+=1
                node.func.id='monitor_predict' if key=='predict' else 'monitor_evaluate'
            return self.generic_visit(node)
    tree=Transform().visit(tree)
    if changes != dict(train_copy=1,train_match=1,train_summary=1,predict=1,evaluate=1,plots=2,stop=1,finished=1):
        raise ValueError('Unexpected scientific engine structure: '+str(changes))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)+'\n'


def monitor_predict(model,loader,device,objective=None):
    from collections import defaultdict
    import torch
    from fmcw_training.engine import inputs,to_device
    samples=[];sums=defaultdict(float);count=0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch=to_device(batch,device);out=model(*inputs(batch,device),return_aux=True)
            if not all(torch.isfinite(out[k]).all() for k in ('clean','presence','logits')):
                raise FloatingPointError('Nonfinite validation output.')
            count+=len(batch['y'])
            if objective:
                loss,parts=objective(out,batch,epoch=100000,warmup=1)
                sums['loss']+=float(loss.sum())
                for key,value in parts.items():sums[key]+=float(value.sum())
            for j,meta in enumerate(batch['meta']):
                samples.append(dict(presence=out['presence'][j].cpu().numpy(),
                    **{k:meta[k] for k in ('range','valid','targets','resolution','group','index','label')}))
    return samples,{k:v/max(1,count) for k,v in sums.items()}


def monitor_evaluate(samples,threshold,confidence=False):
    from fmcw_training.metrics import score_predictions
    if confidence:raise ValueError('No bootstrap during epoch monitoring.')
    return dict(presence=score_predictions(samples,threshold),mean_phase_error_rad=None,mean_nmse=None)


def install(experiment):
    from fmcw_training import engine
    source=fit_source(engine)
    namespace=dict(engine.Experiment.fit.__globals__,monitor_predict=monitor_predict,monitor_evaluate=monitor_evaluate)
    exec(compile(source,'<fixed-budget-public-fit>','exec'),namespace)
    experiment.fit=types.MethodType(namespace['fit'],experiment)
