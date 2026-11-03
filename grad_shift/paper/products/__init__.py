"""Paper products (figures, tables, in-text numbers) and the analyses they need.

Each module in this package defines ``Analysis`` and/or ``Product`` subclasses;
they are discovered automatically (``discover``), so adding a figure never
edits a shared registry. See ``base.py`` for the interfaces.
"""

import importlib
import pkgutil
from typing import Dict, Tuple

from .base import Analysis, Product


def discover() -> Tuple[Dict[str, Analysis], Dict[str, Product]]:
    """Instantiate every Analysis/Product subclass in this package, by name."""
    analyses: Dict[str, Analysis] = {}
    products: Dict[str, Product] = {}
    for mod in pkgutil.iter_modules(__path__):
        if mod.name == "base":
            continue
        m = importlib.import_module(f"{__name__}.{mod.name}")
        for obj in vars(m).values():
            if not isinstance(obj, type) or obj in (Analysis, Product):
                continue
            if issubclass(obj, Analysis) and obj.__module__ == m.__name__:
                inst = obj()
                if inst.name in analyses:
                    raise ValueError(f"duplicate analysis {inst.name}")
                analyses[inst.name] = inst
            elif issubclass(obj, Product) and obj.__module__ == m.__name__:
                inst = obj()
                if inst.name in products:
                    raise ValueError(f"duplicate product {inst.name}")
                products[inst.name] = inst
    return analyses, products
