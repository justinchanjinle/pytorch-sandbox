import torch

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.version.cuda)

    t = torch.rand(10, 10).cuda()
    print(t.device)
