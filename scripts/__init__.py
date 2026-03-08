"""Kintuadi Energy scripts package (import-safe)."""

__version__ = "1.1.1"
__author__ = "Kintuadi Energy Team"

__all__ = []
MODULES_LOADED = False

# Imports opcionais (não quebram o pacote se um módulo faltar no ambiente)
try:
    from .ccee_collector_v2 import CCEEPLDCollector  # type: ignore
    __all__.append("CCEEPLDCollector")
except Exception:
    pass

try:
    from .integrated_collector_v2 import KintuadiIntegratedCollectorV2  # type: ignore
    __all__.append("KintuadiIntegratedCollectorV2")
except Exception:
    pass

try:
    from .ons_collector_v2 import ONSCollectorV2  # type: ignore
    __all__.append("ONSCollectorV2")
except Exception:
    pass

try:
    from .utils import make_serializable, save_json, load_json  # type: ignore
    __all__.extend(["make_serializable", "save_json", "load_json"])
except Exception:
    pass

MODULES_LOADED = bool(__all__)


def get_version():
    return __version__


def get_available_modules():

    return __all__