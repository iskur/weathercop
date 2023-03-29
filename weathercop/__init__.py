try:
    # prefer local configuration
    import cop_conf
except ImportError:
    from . import cop_conf
from . import vine, copulae

