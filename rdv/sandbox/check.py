import typing as _typing
import os
# os.environ['RDV_DEBUG'] = 'True'
import rdv
import torch
import cProfile


class MyMap(rdv.Map):
    __extension_info__ = dict(
        code = """
        FORWARD {{
            for (int i=0; i<OUTPUT_DIM; i++) _output[i] = parameters.a;
        }}
        """,
        parameters=dict(
            a=float,
        ),
        generics=dict(
        )
    )

    def __init__(self, a: float, input_dim=None, output_dim=None):
        super().__init__(INPUT_DIM=input_dim, OUTPUT_DIM=output_dim)
        self.a = a



class Filler(rdv.Compute):
    __extension_info__ = dict(
        parameters=dict(
            t=torch.Tensor,
            a=float
        ),
        code="""
MAIN(tid) {
    // PRINT("Hola %f", parameters.a);
    float_ptr(parameters.t).data[tid.x] = parameters.a + random()*0.1 + random_normal()*0.1 + random_normal()*random();
}
        """
    )

    def bind(self, *args, **kwargs) -> rdv.ComputeTask:
        t, a = args
        task = Filler.create_task(t.numel())
        task.binder.t = rdv.wrap(t, mode='out')
        task.binder.a = a
        return task

    def result(self, task: rdv.ComputeTask) -> _typing.Any:
        task.binder.t.unwrap()

# t = torch.zeros(5000, 300, device='cuda')
t = rdv.tensor(5000, 300)

print(t)
with rdv.time_check("Prewarm"):
    Filler.eval(t, 0.4)


def torch_filler(t: torch.Tensor, a: float):
    t.copy_(torch.rand_like(t)*0.1 + torch.randn_like(t)*0.1 + a + torch.randn_like(t)*torch.rand_like(t))


import numpy as np

# torch_filler(t, np.random.rand())


def test():
    with rdv.time_check("Eval from cache"):
        for i in range(10000):
            # torch_filler(t, np.random.rand())
            # torch.cuda.synchronize()
            # torch.cuda.empty_cache()
            Filler.eval(t, np.random.rand())
            # t.copy_(t_vk)
            # t[0,0].item()
        print(t[0,0].item())
    print(t)

cProfile.run('test()', sort='cumtime')
# test()

# m = MyMap(a=0.2, input_dim=3, output_dim=4)
# print(m(torch.zeros(4, 3)))




