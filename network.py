import torch
import torch.nn as nn


class CartPoleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, state):
        # Raw network output
        x = self.network(state)

        # Bound the physical force to [-10, +10] N
        force = 10.0 * torch.tanh(x)

        return force


if __name__ == "__main__":
    model = CartPoleNet()

    # Example state:
    # [cart position, pole angle, cart velocity, pole angular velocity]
    state = torch.tensor([
        [0.0, 0.05, 0.0, 0.0]
    ], dtype=torch.float32)

    force = model(state)

    print("Network:")
    print(model)

    print("\nInput state:")
    print(state)

    print("\nOutput force:")
    print(force)

    print("\nNumber of parameters:")
    print(sum(p.numel() for p in model.parameters()))