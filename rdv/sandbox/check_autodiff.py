import typing
import torch
import os
# os.environ['RDV_DEBUG'] = 'True'
import rdv
import cProfile


class MyMap(rdv.Map):
    __extension_info__ = dict(
        code = """
        FORWARD {
            Tensor t = load_deferred(parameters.t);
            float_ptr t_data = float_ptr(t.data_ptr);
            for (int i=0; i<INPUT_DIM; i++) _output[i] = _input[i] + t_data.data[i];
        }
        
        BACKWARD {
            Tensor t = load_deferred(parameters.t);
            float_ptr t_data = float_ptr(t.data_ptr);
            float_ptr t_grad = float_ptr(t.grad_ptr);
            for (int i=0; i<INPUT_DIM; i++) {
            #ifdef INPUT_REQUIRES_GRAD
                _input_grad[i] += _output_grad[i];
            #endif
                if (t.grad_ptr != 0)
                    t_grad.data[i] += _output_grad[i];
            }
        }
        """,
        parameters=dict(
            t=rdv.DeferrableField
        ),
        generics=dict(
        )
    )

    def __init__(self, t: typing.Union[rdv.deferred, torch.Tensor],
                 **kwargs):
        super().__init__(**kwargs)
        self.t = rdv.ensure_tensor(t, map_dim=1)

    def clone(self, **kwargs) -> 'MyMap':
        return MyMap(self.t, **kwargs)


t = rdv.tensor(3, requires_grad=False)
# t = torch.zeros(4, device='cuda')
with torch.no_grad():
    t.copy_(torch.tensor([0.2, 0.3, 0.4]))

# print(t.device_ptr)

m = MyMap(
    # t,
    rdv.deferred('tensor'),
    input_dim=3, output_dim=3).cast(input_requires_grad=True)

cpu_tensor = torch.rand(500000, 3, requires_grad=True)

input_tensor = rdv.tensor_clone(cpu_tensor)

# input_tensor.backward(torch.ones_like(input_tensor, device=input_tensor.device))
#
# print(input_tensor.grad)
# exit()
# input_tensor = rdv.tensor_clone(input_tensor)

output_tensor = m(input_tensor, tensor=t)

output_tensor.sum().backward()

print(output_tensor)
# exit()
