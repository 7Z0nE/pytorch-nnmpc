import control
import torch


def rosebrock(x):
    return (1 - x[0])**2 + 100(x[1] - x[0]**2)**2


def test_cem_optimizer_2dparabel():
    f = lambda x: torch.tensor([torch.sum(-x**2) for x in x])
    start = torch.tensor([[3., 3.], [3., 3.]])
    minimum = control.cem_optimize(start, torch.tensor([[5.,5.], [5.,5.]]), f, (2, 2), 20)
    err = torch.mean(minimum - torch.tensor([[0., 0.]]))
    print("Approximated 2dparabel with err ", err)

if __name__=="__main__":
    test_cem_optimizer_2dparabel()