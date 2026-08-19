# import os
# os.environ['RDV_DEBUG'] = 'True'
import rdv
import torch


c1 = rdv.Map.as_map(1) #.cast(input_dim=3)
c2 = rdv.IdentityMap.get_instance()
c3 = c1 + c2
c3 = c3.cast(input_dim=3, output_dim=3).then(rdv.IdentityMap.get_instance() + rdv.Map.as_map(2.0))

t = torch.rand(5, 3)

print(t)
print(c3(t))
