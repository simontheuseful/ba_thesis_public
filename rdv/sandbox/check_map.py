import typing
import typing as _typing

import torch
import os
# os.environ['RDV_DEBUG'] = 'True'
import rdv
import cProfile


# class MapEval(rdv.Compute):
    # __extension_info__ = dict(
    #     path='../src/rdv/include/system/map_eval.h',
    #     parameters=dict(
    #         input_tensor=torch.Tensor,
    #         output_tensor=torch.Tensor,
    #         map=rdv.Map
    #     )
    # )
    #
    # def bind(self, *args, **kwargs) -> rdv.ComputeTask:
    #     map, input = args
    #     assert not map.is_generic
    #     input_dim = input.shape[-1]
    #     output_dim = map.output_dim
    #     output = rdv.tensor(*input.shape[:-1], output_dim)
    #     assert map.input_dim == input_dim
    #     task = MapEval.create_task(input.numel() // input_dim, map, MAP_INPUT_DIM=input_dim, MAP_OUTPUT_DIM=output_dim)
    #     binder = task.binder
    #     binder.input_tensor = rdv.wrap(input, 'in')
    #     binder.output_tensor = rdv.wrap(output, 'out')
    #     binder.map = map
    #     return task
    #
    # def result(self, compute_task: rdv.ComputeTask) -> _typing.Any:
    #     compute_task.binder.input_tensor.unwrap()
    #     return compute_task.binder.output_tensor.unwrap()

def eval_map(map: rdv.Map, input: torch.Tensor, **deferred_parameters):
    # return MapEval.eval(map, input, deferred_parameters=deferred_parameters)
    return map(input, **deferred_parameters)

class MyMap(rdv.Map):
    __extension_info__ = dict(
        code = """
        FORWARD {
            Tensor t = load_deferred(parameters.t);
            float_ptr t_data = float_ptr(t.data_ptr);
            for (int i=0; i<OUTPUT_DIM; i++) _output[i] = parameters.a + t_data.data[i];
        }
        """,
        parameters=dict(
            t=rdv.DeferrableField,
            a=float,
        ),
        generics=dict(
        )
    )

    def __init__(self, t: typing.Union[rdv.deferred, torch.Tensor], a: float, input_dim=None, output_dim=None):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.t = rdv.ensure_tensor(t, map_dim=1)
        self.a = a


t = rdv.tensor(4)
# t = torch.zeros(4, device='cuda')
t.copy_(torch.tensor([0.2, 0.3, 0.4, 0.5]))

# print(t.device_ptr)

m = MyMap(
    # t,
    rdv.deferred('tensor'),
    0.3, 3, 4)

cpu_tensor = torch.rand(500000, 3, requires_grad=False)

input_tensor = rdv.tensor_clone(cpu_tensor)

# input_tensor.backward(torch.ones_like(input_tensor, device=input_tensor.device))
#
# print(input_tensor.grad)
# exit()
# input_tensor = rdv.tensor_clone(input_tensor)

output_tensor = eval_map(m, input_tensor, tensor=t)
print(output_tensor)
# exit()
def test():
    for i in range(10000):
        # torch_filler(t, np.random.rand())
        # torch.cuda.synchronize()
        # torch.cuda.empty_cache()
        eval_map(m, input_tensor, tensor=t)
        # t.copy_(t_vk)
        # t[0,0].item()

cProfile.run('test()', sort='cumtime')