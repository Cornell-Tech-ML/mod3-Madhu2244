from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """Modules form a tree that store parameters and other submodules. They make up the basis of neural network stacks."""

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module."""
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """Set the `training` flag of this and descendent to true."""
        self.training = True
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        """Set the `training` flag of this and descendent to false."""
        self.training = False
        for module in self.modules():
            module.eval()

    def named_parameters(self, prefix: str = "") -> Sequence[Tuple[str, Parameter]]:
        """Collect all the parameters of this module and its descendents."""
        results = []
        for key, value in self._parameters.items():
            results.append((prefix + key, value))
        for key, value in self._modules.items():
            results += value.named_parameters(prefix + key + ".")
        return results

    def parameters(self) -> Sequence[Parameter]:
        """Enumerate over all the parameters of this module and its descendents."""
        results = []
        for value in self._parameters.values():
            results.append(value)
        for value in self.modules():
            results += value.parameters()
        return results

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
        ----
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
        -------
            Newly created parameter.

        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        """Set the attribute of the module."""
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        """Get the attribute of the module."""
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the module."""
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        """Return the representation of the module."""

        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        """Initialize the parameter."""
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value."""
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        """Return the representation of the parameter."""
        return repr(self.value)

    def __str__(self) -> str:
        """Return the string representation of the parameter."""
        return str(self.value)
