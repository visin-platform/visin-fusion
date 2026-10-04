"""Dataset URI hooks. Plain filesystem roots need no integration."""

from importlib import import_module

from visin_fusion.providers import HUB

_resolvers = {
    "visin": "visin_fusion.data.visin_datasets:resolve_root",
    HUB: "visin_fusion.data.hf_datasets:resolve_root",
}


def register_resolver(prefix, resolver):
    """Register a callable for ``prefix:name`` roots."""
    _resolvers[prefix] = resolver


def resolve_root(root):
    """The folder a ``Dataset.dataset_root`` names.

    ``prefix:name`` goes to the resolver registered for the prefix (``visin:zod`` and
    ``hf:org/name@commit`` download); anything else is
    returned as it is.
    """
    if not isinstance(root, str) or ":" not in root:
        return root
    prefix = root.split(":", 1)[0]
    resolver = _resolvers.get(prefix)
    if resolver is None:
        return root
    if isinstance(resolver, str):
        module, name = resolver.split(":", 1)
        resolver = getattr(import_module(module), name)
    return resolver(root)
