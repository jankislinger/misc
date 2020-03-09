import math
from typing import Dict, List


class Expr:

    def eval(self, **kwargs) -> float:
        raise NotImplementedError()

    def derive(self, wrt: str) -> 'Expr':
        raise NotImplementedError()

    def prune(self) -> 'Expr':
        raise NotImplementedError()

    def maximize(self, start: Dict[str, float], learning_rate, other_vars=None, tol=1e-12):
        if other_vars is None:
            other_vars = {}
        derivations = {var_name: self.derive(wrt=var_name).prune() for var_name in start}
        x = start.copy()

        delta = {k: e.eval(**x, **other_vars) for k, e in derivations.items()}
        niter = 0
        while max(abs(d) for d in delta.values()) > tol:
            # print(niter, x, delta, self.eval(**x, **other_vars))
            for var_name in x.keys():
                x[var_name] += delta[var_name] * learning_rate
            delta = {k: e.eval(**x, **other_vars) for k, e in derivations.items()}
            niter += 1

        print('niter =', niter)

        y = self.eval(**x, **other_vars)
        return y, x

    def minimize(self, start, learning_rate, other_vars=None, tol=1e-12):
        y, x = Times(self, -1).maximize(start, learning_rate, other_vars, tol)
        return -y, x

    def derive_n(self, wrt, n):
        derivative = self
        for _ in range(n):
            derivative = derivative.derive(wrt).prune()
        return derivative

    def __repr__(self):
        return str(self)

    def __mul__(self, other):
        return Times(self, other)

    def __add__(self, other):
        return Add(self, other)

    def __pow__(self, power, modulo=None):
        return Pow(self, power)

    def __sub__(self, other):
        return Add(self, Const(-1) * other)

    def __truediv__(self, other):
        return Div(self, other)


class Operator(Expr):
    right: Expr
    left: Expr
    operator = ""

    def __init__(self, left, right):
        self.right = ensure_expr(right)
        self.left = ensure_expr(left)

    def eval(self, **kwargs):
        return self.fun(self.left.eval(**kwargs), self.right.eval(**kwargs))

    def derive(self, wrt: str):
        raise NotImplementedError()

    def prune(self):
        left = self.left.prune()
        right = self.right.prune()

        if isinstance(left, Const) and isinstance(right, Const):
            return Const(self.fun(left.value, right.value))
        return type(self)(left, right)

    def fun(self, x, y):
        raise NotImplementedError()

    def __str__(self):
        return f"({self.left} {self.operator} {self.right})"


class Function(Expr):
    child: Expr
    name = 'f'

    def __init__(self, child):
        self.child = ensure_expr(child)

    def eval(self, **kwargs):
        return self.fun(self.child.eval(**kwargs))

    def derive(self, wrt: str):
        raise NotImplementedError()

    def prune(self) -> 'Expr':
        child = self.child.prune()
        if isinstance(child, Const):
            return Const(self.fun(child.value))
        return self

    def fun(self, x):
        raise NotImplementedError()

    def __str__(self):
        return f"{self.name}({self.child})"


class Const(Expr):
    value: float

    def __init__(self, value):
        self.value = value

    def eval(self, **kwargs):
        return self.value

    def derive(self, wrt: str):
        return Const(0)

    def prune(self) -> 'Expr':
        return self

    def __str__(self):
        return str(self.value)


class Variable(Expr):
    name: str

    def __init__(self, name):
        self.name = name

    def eval(self, **kwargs):
        return kwargs[self.name]

    def derive(self, wrt: str):
        return Const(1 if wrt == self.name else 0)

    def prune(self) -> 'Expr':
        return self

    def __str__(self):
        return self.name


class Add(Operator):
    operator = '+'

    def fun(self, x, y):
        return x + y

    def derive(self, wrt: str):
        return self.left.derive(wrt) + self.right.derive(wrt)

    def prune(self):
        left = self.left.prune()
        right = self.right.prune()

        if isinstance(left, Const) and left.value == 0.0:
            return right
        if isinstance(right, Const) and right.value == 0.0:
            return left
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(self.fun(left.value, right.value))
        return type(self)(left, right)


class Times(Operator):
    operator = '*'

    def fun(self, x, y):
        return x * y

    def derive(self, wrt: str):
        return self.left.derive(wrt) * self.right + self.left * self.right.derive(wrt)

    def prune(self):
        left = self.left.prune()
        right = self.right.prune()

        if isinstance(left, Const) and left.value == 0:
            return Const(0)
        if isinstance(right, Const) and right.value == 0:
            return Const(0)
        if isinstance(left, Const) and left.value == 1:
            return right
        if isinstance(right, Const) and right.value == 1:
            return left
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(self.fun(left.value, right.value))
        return type(self)(left, right)


class Div(Operator):
    operator = '/'

    def eval(self, **kwargs):
        return self.fun(self.left.eval(**kwargs), self.right.eval(**kwargs))

    def fun(self, x, y):
        return x / y

    def derive(self, wrt: str):
        return (self.left.derive(wrt) * self.right - self.left * self.right.derive(wrt)) / (self.right ** 2)


class Pow(Operator):
    operator = '**'

    def fun(self, x, y):
        return x ** y

    def derive(self, wrt: str):
        power = self.right * self.left ** (self.right - Const(1)) * self.left.derive(wrt)
        exp = Log(self.right) * self.left ** self.right * self.right.derive(wrt)
        return power + exp

    def prune(self):
        left = self.left.prune()
        right = self.right.prune()

        if isinstance(right, Const):
            if right.value == 0:
                return Const(1)
            if right.value == 1:
                return left
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(self.fun(left.value, right.value))
        return type(self)(left, right)


class Log(Function):
    name = 'log'

    def fun(self, x):
        return math.log(x)

    def derive(self, wrt: str):
        return self.child.derive(wrt) / self.child


class Cos(Function):
    name = 'cos'

    def fun(self, x):
        return math.cos(x)

    def derive(self, wrt: str):
        return Const(-1) * Sin(self.child) * self.child.derive(wrt)


class Sin(Function):
    name = 'sin'

    def fun(self, x):
        return math.sin(x)

    def derive(self, wrt: str):
        return Cos(self.child) * self.child.derive(wrt)


class Square(Function):
    name = 'square'

    def fun(self, x):
        return x ** 2

    def derive(self, wrt: str):
        return Const(2) * self.child * self.child.derive(wrt)


class Sum(Expr):
    children: List[Expr]

    def __init__(self, children):
        self.children = [ensure_expr(c) for c in children]

    def eval(self, **kwargs) -> float:
        return sum(c.eval(**kwargs) for c in self.children)

    def derive(self, wrt: str) -> 'Expr':
        return Sum([c.derive(wrt) for c in self.children])

    def prune(self) -> 'Expr':
        children = [c.prune() for c in self.children]
        if all(isinstance(c, Const) for c in children):
            return Const(sum(c.value for c in children))
        return Sum(children)

    def __str__(self):
        return " + ".join(str(c) for c in self.children)


def ensure_expr(x):
    if isinstance(x, Expr):
        return x
    if isinstance(x, str):
        return Variable(x)
    if isinstance(x, int) or isinstance(x, float):
        return Const(x)
    raise TypeError(f"Unknown type: {type(x)}")


if __name__ == '__main__':
    print(Pow('x', 2).derive_n('x', 2))
    print(Const(2).derive('x'))
    # expr = Variable('x') - Pow('x', 2)
    # print(expr)
    # print(expr.derive('x').prune())
    # print(expr.eval(x=1))
    # t = time.time()
    # y, x = expr.maximize('x', -1, 1e-3)
    # t = time.time() - t
    # print(t * 1000, 'ms')
    # print(f"Maximum of {y} found at {x}")

    expr = Sin(Pow('x', 2) * Variable('y')) - Variable('x')
    print(expr.maximize({'x': 0}, 1e-3, {'y': 2}))

    print(Log(Sin(2)).prune())

    print(Sum([Const(i) for i in range(5)]))
    print(Sum([Const(i) for i in range(5)]).prune())
