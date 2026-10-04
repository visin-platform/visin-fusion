"""The ids of the stores this project reads from and reports to, as Visin and the config spell them.

Nothing else spells one: a module that names a store imports its id from here, so a second store is one more
constant and the registries that key on it (dataset resolvers, data evidence), not a search for string literals.
"""

from __future__ import annotations

HUB = "hf"
