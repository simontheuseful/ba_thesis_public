import unittest
import torch
import os
# os.environ['RDV_DEBUG'] = 'True'
import rdv


def sample2d_torch(grid, points):
    o = torch.nn.functional.grid_sample(
        grid.permute(2, 0, 1).unsqueeze(0),  # (1, C, H, W)
        points.unsqueeze(0).unsqueeze(2),      # (1, N, 1, 2)
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    o = o.squeeze(0).squeeze(2).permute(1, 0)  # (C, N)
    return o

def sample3d_torch(grid, points):
    o = torch.nn.functional.grid_sample(
        grid.permute(3, 0, 1, 2).unsqueeze(0),  # (1, C, D, H, W
        points.unsqueeze(0).unsqueeze(0).unsqueeze(0),      # (1, N, 1, 3)
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    o = o.squeeze(0).squeeze(2).squeeze(1).permute(1, 0)  # (C, N)
    return o



class TestExample(unittest.TestCase):
    def test_sample3d_forward(self):
        grid = torch.rand(64, 64, 64, 8, device='cuda')  # A 3D grid with shape (D, H, W, 8)
        sampler = rdv.Sample3DMap(grid)
        input = torch.rand(1000, 3, device='cuda') * 3 - 1.5
        result_map = sampler(input)
        expected = sample3d_torch(grid, input)
        self.assertTrue(torch.allclose(result_map, expected, atol=1e-6))

    def test_sample3d_backward(self):
        sampler = rdv.Sample3DMap(rdv.deferred('grid'), output_dim=4)
        grid = torch.rand(32, 32, 32, 4, device='cuda') # A 3D grid with shape (D, H, W, 4)
        grid = rdv.tensor_clone(grid)
        grid.requires_grad_(True)
        input = torch.rand(500, 3, device='cuda') * 3 - 1.5
        result = sampler(input, grid=grid)
        random_grads = torch.randn_like(result)
        result.backward(random_grads)
        computed_grads = grid.grad.clone()
        grid.grad = None
        expected = sample3d_torch(grid, input)
        expected.backward(random_grads)
        expected_grads = grid.grad.clone()
        self.assertTrue(torch.allclose(computed_grads, expected_grads, atol=1e-6))

    def test_sample2d_forward(self):
        grid = torch.rand(64, 64, 8, device='cuda')  # A 2D grid with shape (H, W, 8)
        sampler = rdv.Sample2DMap(grid)
        input = torch.rand(1000, 2, device='cuda') * 2 - 1.0
        result_map = sampler(input)
        expected = sample2d_torch(grid, input)
        self.assertTrue(torch.allclose(result_map, expected, atol=1e-6))

    def test_sample2d_backward(self):
        sampler = rdv.Sample2DMap(rdv.deferred('grid'), output_dim=4)
        grid = torch.rand(32, 32, 4, device='cuda') # A 2D grid with shape (H, W, 4)
        grid = rdv.tensor_clone(grid)
        grid.requires_grad_(True)
        input = torch.rand(500, 2, device='cuda') * 2 - 1.0
        result = sampler(input, grid=grid)
        random_grads = torch.randn_like(result)
        result.backward(random_grads)
        computed_grads = grid.grad.clone()
        grid.grad = None
        expected = sample2d_torch(grid, input)
        expected.backward(random_grads)
        expected_grads = grid.grad.clone()
        self.assertTrue(torch.allclose(computed_grads, expected_grads, atol=1e-6))

    def test_constant_forward(self):
        c = rdv.as_map([1.0, 2.0, 3.0]).cast(input_dim=2)
        input = torch.rand(1000, 2, device='cuda')
        result = c(input)
        expected = torch.tensor([1.0, 2.0, 3.0], device='cuda').unsqueeze(0).repeat(1000, 1)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_constant_backward(self):
        c = rdv.ConstantMap(t=rdv.deferred('value'), input_dim=2, output_dim=3)
        input = torch.rand(500, 2, device='cuda')
        param = torch.tensor([1.0, 2.0, 3.0], device='cuda', requires_grad=True)
        result = c(input, value=param)
        random_grads = torch.randn_like(result)
        result.backward(random_grads)
        self.assertIsNotNone(param.grad)
        computed_grads = param.grad.clone()
        param.grad = None
        expected = torch.tensor([1.0, 2.0, 3.0], device='cuda', requires_grad=True)
        result = expected.unsqueeze(0).repeat(500, 1)
        result.backward(random_grads)
        expected_grads = expected.grad.clone()
        self.assertTrue(torch.allclose(computed_grads, expected_grads, atol=1e-6))

    def template_map_op(self, op, input_requires_grad=False, parameter_requires_grad=False):
        p1 = torch.tensor([3.0, 4.0], device='cuda', requires_grad=parameter_requires_grad)
        p2 = torch.tensor([10.0, 20.0], device='cuda', requires_grad=parameter_requires_grad)
        m1 = rdv.ConstantMap(rdv.deferred('p1'), input_dim=2, output_dim=2)
        m2 = rdv.ConstantMap(rdv.deferred('p2'), input_dim=2, output_dim=2)
        if op == 'add':
            m3 = m1 + m2
        elif op == 'sub':
            m3 = m1 - m2
        elif op == 'mul':
            m3 = m1 * m2
        elif op == 'div':
            m3 = m1 / m2
        else:
            raise ValueError(f"Unsupported operation: {op}")
        m1 = m1.cast(input_requires_grad=input_requires_grad)
        m2 = m2.cast(input_requires_grad=input_requires_grad)
        m3 = m3.cast(input_requires_grad=input_requires_grad)
        input = torch.rand(1000, 2, device='cuda', requires_grad=input_requires_grad)
        result = m3(input, p1=p1, p2=p2)
        if op == 'add':
            expected = m1(input, p1=p1) + m2(input, p2=p2)
        elif op == 'sub':
            expected = m1(input, p1=p1) - m2(input, p2=p2)
        elif op == 'mul':
            expected = m1(input, p1=p1) * m2(input, p2=p2)
        elif op == 'div':
            expected = m1(input, p1=p1) / m2(input, p2=p2)
        else:
            raise ValueError(f"Unsupported operation: {op}")

        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

        if input_requires_grad or parameter_requires_grad:
            random_grads = torch.randn_like(result)
            result.backward(random_grads)
            if input_requires_grad:
                computed_grads = input.grad.clone()
                input.grad = None
            if parameter_requires_grad:
                computed_param_grads = []
                for p in [p1, p2]:
                    computed_param_grads.append(p.grad.clone())
                    p.grad = None
            expected.backward(random_grads)
            if input_requires_grad:
                expected_grads = input.grad.clone()
                self.assertTrue(torch.allclose(computed_grads, expected_grads, atol=1e-6))
            if parameter_requires_grad:
                for i, p in enumerate([p1, p2]):
                    expected_param_grads = p.grad.clone()
                    self.assertTrue(
                        torch.allclose(computed_param_grads[i], expected_param_grads, atol=1e-6),
                        msg=f"Parameter gradient mismatch for parameter {i} in operation {op} with input_requires_grad={input_requires_grad} and parameter_requires_grad={parameter_requires_grad}"
                    )

    def test_map_ops(self):
        for op in ['add', 'sub', 'mul', 'div']:
            for input_grad in [False, True]:
                for param_grad in [False, True]:
                    self.template_map_op(op, input_requires_grad=input_grad, parameter_requires_grad=param_grad)




if __name__ == '__main__':
    unittest.main()