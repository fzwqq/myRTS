import torch

fc = torch.nn.Linear(2, 2, bias=False)
with torch.no_grad():
    fc.eval()
    fc.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

fc_gpu = fc.to("cuda")
fc_gpu.eval()

fc_gpu.weight.requires_grad = False

def wrapper(inp):
    return fc_gpu(inp.to("cuda").detach()).to("cpu")

with torch.no_grad():
    traced_for_cpu = torch.jit.trace(wrapper, torch.tensor([10.0, 100.0]))

traced_for_cpu.save("model.pt2")
