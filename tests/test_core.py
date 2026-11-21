import unittest
import torch
from oodsteer.steering import nci_gradient, intermediate_layers, USteer
from oodsteer.data import extract

class CoreTests(unittest.TestCase):
    def test_gradient_against_autograd(self):
        torch.manual_seed(7)
        for eps in (1e-6, .1):
            x = torch.randn(2,3,19,requires_grad=True)
            gain = torch.randn(19)
            w = torch.randn(2,3,19)
            z = gain*x/(x.square().mean(-1,keepdim=True)+eps).sqrt()
            score = (w*z).sum(-1).square()/z.square().sum(-1)
            expected, = torch.autograd.grad(score.sum(),x)
            actual = nci_gradient(x.detach(),gain,eps,w)
            torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-5)

    def test_layers(self):
        self.assertEqual(intermediate_layers(16,3),[12,13,14])
        self.assertEqual(intermediate_layers(36,8),list(range(27,35)))

    def test_extraction(self):
        self.assertEqual(extract('The answer is $1,200.00.', 'gsm8k'),'1.2E+3')
        self.assertEqual(extract('So the answer is (c).','csqa'),'C')
        self.assertEqual(extract('So the answer is no.','strategyqa'),'no')
        self.assertIsNone(extract('I think C','csqa'))

    def test_hook_and_cleanup(self):
        from types import SimpleNamespace
        norm = SimpleNamespace(weight=torch.ones(7),variance_epsilon=1e-6)
        blocks = [torch.nn.Identity() for _ in range(4)]
        head = torch.nn.Linear(7,13,bias=False)
        model = SimpleNamespace(model=SimpleNamespace(norm=norm,layers=blocks),get_output_embeddings=lambda:head)
        x = torch.randn(1,5,7)
        with USteer(model,[1],.3) as steer:
            y = blocks[1](x)
            torch.testing.assert_close(y[:,:-1],x[:,:-1])
            self.assertAlmostEqual((y[:,-1]-x[:,-1]).norm().item(),.3,places=5)
            self.assertEqual(steer.calls,1)
        torch.testing.assert_close(blocks[1](x),x)

if __name__ == '__main__':
    unittest.main()
