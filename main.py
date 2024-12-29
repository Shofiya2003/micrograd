from data import Value

x1 = Value(2.0)
x2 = Value(0.0)

# Weights
w1 = Value(-3.0)
w2 = Value(1.0)

# biases
b = Value(6.7)

x1w1 = x1*w1
x2w2 = x2*w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b
o = n.tanh()
print(o)

o.backward()