__all__ = [
    "copulae",
    "cop_conf",
    "example_data",
    "multisite",
    "multisite_conditional",
    "find_copula",
    "plotting",
    "seasonal_cop",
    "tools",
    "vine",
]

try:
    # prefer local configuration
    import cop_conf
except ImportError:
    from . import cop_conf

# Module-level lazy loading to defer imports of heavy dependencies.
# This allows importing copulae for ufunc generation without pulling in vine
# and its dependencies (networkx, cartopy, etc).
_lazy_modules = {
    "vine": ".vine",
    "copulae": ".copulae",
    "example_data": ".example_data",
    "multisite": ".multisite",
    "multisite_conditional": ".multisite",
    "find_copula": ".copulae",
    "plotting": ".plotting",
    "seasonal_cop": ".seasonal_cop",
    "tools": ".tools",
}


def __getattr__(name):
    """Lazily import modules and extract attributes as needed."""
    if name in _lazy_modules:
        import importlib
        module_name = _lazy_modules[name]
        module = importlib.import_module(module_name, __name__)
        # For function attributes like find_copula or multisite, extract from module
        if name in ("find_copula", "multisite", "multisite_conditional"):
            return getattr(module, name)
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include lazy modules in dir() output."""
    return list(__all__)
