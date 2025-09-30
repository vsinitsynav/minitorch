from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    return (
        f(*vals[:arg], vals[arg] + epsilon, *vals[arg + 1 :])
        - f(*vals[:arg], vals[arg] - epsilon, *vals[arg + 1 :])
    ) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative `x` into the variable's derivative value."""
        pass

    @property
    def unique_id(self) -> Any:
        """A unique identifier for this variable."""
        pass

    def is_leaf(self) -> Any:
        """True if this variable created by the user (no `last_fn`)"""
        pass

    def is_constant(self) -> Any:
        """True if this variable is a constant (not a result of a computation)"""
        pass

    @property
    def parents(self) -> Any:
        """The parent Variables used to compute this Variable."""
        pass

    def chain_rule(self, d_output: Any) -> Any:
        """Given the derivative of some later variable with respect to this variable,
        compute the local gradient and return the variable and its gradient.
        """
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    used = dict()
    order: List[Variable] = []

    def dfs(v: Variable) -> None:
        if v.unique_id in used:
            return
        used[v.unique_id] = 1
        for parent in v.parents:
            dfs(parent)
        order.append(v)

    dfs(variable)
    return reversed(order)


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable : The right-most variable
        deriv : Its derivative that we want to propagate backward to the leaves.

    Returns:
    -------
    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    derivs = dict()
    derivs[variable.unique_id] = deriv
    for v in topological_sort(variable):
        d = derivs[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(d)
        else:
            for child, child_deriv in v.chain_rule(d):
                if child.unique_id not in derivs:
                    tmp = child_deriv
                else:
                    tmp = derivs[child.unique_id] + child_deriv
                derivs[child.unique_id] = tmp


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved values."""
        return self.saved_values
