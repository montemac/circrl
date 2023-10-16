import torch as t
from torch import nn
from circrl.hooks import HookManager


def test_hook_manager():
    # Define a simple model to test with
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(
                16, 32, kernel_size=3, padding=1, bias=False
            )
            self.relu2 = nn.ReLU(inplace=True)
            self.fc = nn.Linear(32 * 8 * 8, 10, bias=False)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
            x = x.view(-1, 32 * 8 * 8)
            x = self.fc(x)
            return x

    # Create a test input tensor
    input_tensor = t.randn(1, 3, 32, 32)

    # Test caching activations
    model = TestModel()
    with HookManager(model, cache=["relu1", "relu2"]) as cache_results:
        output = model(input_tensor)
        assert "relu1" in cache_results
        assert "relu2" in cache_results
        assert cache_results["relu1"].shape == (1, 16, 32, 32)
        assert cache_results["relu2"].shape == (1, 32, 32, 32)

    # Test patching activations
    with HookManager(model, patch={"relu1": t.zeros(16, 32, 32)}) as _:
        output = model(input_tensor)
        assert (output == 0).all()

    # Test applying hook functions
    def double_output_hook(input, output):
        return output * 2

    single_output = model(input_tensor)
    with HookManager(model, hook={"relu1": double_output_hook}) as _:
        output = model(input_tensor)
        assert t.allclose(output, 2 * single_output)
