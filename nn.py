from engine import MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]  # desired targets

n = MLP(3, [4, 4, 1])

for i in range(20):

    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, ypred)])
    for param in n.parameters():
        param.grad = 0

    # backward pass
    loss.backward()

    # updating the parameters
    for p in n.parameters():
        p.data+= -0.05  * p.grad
    print(f"iteration: {i},  loss {loss}")