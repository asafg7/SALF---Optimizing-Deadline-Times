import torch
import torch.nn as nn
import time


class DeadlineConstrainedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.deadline = 0  # deadline in seconds
        self.hooks = []
        self.start_time = time.time()

        # Register hooks for each module
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Module):  # Only hook modules
                pre_hook = module.register_full_backward_pre_hook(self.backward_pre_hook)
                self.hooks.append(pre_hook)
                post_hook = module.register_full_backward_hook(self.backward_post_hook)
                self.hooks.append(post_hook)

    def set_start_time(self, deadline):
        self.start_time = time.perf_counter()
        self.deadline = deadline

    def backward_pre_hook(self, module, grad_output):
        elapsed_time = time.perf_counter() - self.start_time
        #print("Elapsed time:" + str(elapsed_time))
        if elapsed_time > self.deadline:
            print(f"Warning: Module {module.__class__.__name__} exceeded the deadline ({elapsed_time:.4f}s > {self.deadline}s)")
            new_grad_output = (None, ) * len(grad_output)
        else:
            new_grad_output = grad_output
        return new_grad_output

    def backward_post_hook(self, module, grad_input, grad_output):
        elapsed_time = time.perf_counter() - self.start_time
        #print("Elapsed time:" + str(elapsed_time))
        if elapsed_time > self.deadline:
            print(f"Warning: Module {module.__class__.__name__} exceeded the deadline ({elapsed_time:.4f}s > {self.deadline}s)")
            new_grad_input = (None, ) * len(grad_input)
        else:
            new_grad_input = grad_input
        return new_grad_input

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# Example usage
if __name__ == "__main__":
    # Simple MLP for demonstration
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Instantiate model and wrap with deadline constraint
    model = SimpleMLP()
    deadline_model = DeadlineConstrainedModel(model, deadline=0.001)  # 1ms deadline

    # Dummy data
    input_data = torch.randn(64, 784)  # Batch of 64 samples
    target = torch.randint(0, 10, (64,))  # Random targets
    criterion = nn.CrossEntropyLoss()

    # Training step
    optimizer = torch.optim.SGD(deadline_model.parameters(), lr=0.01)

    for epoch in range(5):
        optimizer.zero_grad()
        output = deadline_model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # Remove hooks when done
    deadline_model.remove_hooks()
