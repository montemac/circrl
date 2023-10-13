"""Classes and functions to simplify the use of pytorch hooks for
arbitrary models, specifically for caching and patching activations, and
applying arbitrary hook functions safely."""

from typing import Optional, List, Dict
import torch as t

class HookManager():

    def __init__(self, cache: Optional[List[str]]=None, patch: Optional[Dict[str, ]])