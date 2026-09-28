"""Dataset URI hooks. Plain filesystem roots need no integration."""
from importlib import import_module

_resolvers = {'visin': 'visin_fusion.data.visin_datasets:resolve_root'}


def register_resolver(prefix, resolver):
    """Register a callable for ``prefix:name`` roots."""
    _resolvers[prefix] = resolver


def resolve_root(root):
    if not isinstance(root, str) or ':' not in root:
        return root
    prefix = root.split(':', 1)[0]
    resolver = _resolvers.get(prefix)
    if resolver is None:
        return root
    if isinstance(resolver, str):
        module, name = resolver.split(':', 1)
        resolver = getattr(import_module(module), name)
    return resolver(root)
