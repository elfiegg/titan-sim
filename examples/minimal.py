"""Framework-free synthetic workload illustrating the capture adapter contract."""
from titan_sim.core import Node, Storage
from titan_sim.workload import Workload


def capture(config):
    return Workload(
        nodes=[Node(1, 'producer', 0, attrs={'flops':2000}),
               Node(2, 'consumer', 0, deps=[1], attrs={'flops':3000})],
        storage=[Storage('shared_backing', 0, 4096, producer=1, consumers=[2])],
        manifest={'ranks':[0], 'useful_tokens':0,
                  'scope':'synthetic two-operation example, not a captured model',
                  'unresolved':['Synthetic analytical costs; implementation workspaces are unspecified']},
        hardware=config['hardware'],
    )
