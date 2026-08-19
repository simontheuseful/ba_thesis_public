import unittest
import torch
import rdv


class TestExample(unittest.TestCase):
    def test_map_addition(self):
        # Test addition of two maps
        c1 = rdv.Map.as_map(1)
        c2 = rdv.IdentityMap.get_instance()
        c3 = c1 + c2

        self.assertIsNotNone(c3)

    def test_cast_and_then_operations(self):
        # Test chaining operations with cast and then
        c1 = rdv.Map.as_map(1)
        c2 = rdv.IdentityMap.get_instance()
        c3 = c1 + c2
        c3 = c3.cast(input_dim=3, output_dim=3).then(rdv.IdentityMap.get_instance() + rdv.Map.as_map(2.0))

        t = torch.rand(5, 3)
        result = c3(t)

        self.assertEqual(result.shape, t.shape)

    def test_identity_map(self):
        # Test the identity map functionality
        identity = rdv.IdentityMap.get_instance().cast(input_dim=3)
        t = torch.rand(5, 3)
        result = identity(t).to(t.device)

        self.assertTrue(torch.equal(result, t))


if __name__ == '__main__':
    unittest.main()