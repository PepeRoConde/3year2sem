# Importar la clase base Behaviour
from .behaviour import Behaviour

# Importar los comportamientos específicos
from .avoid_fall import AvoidFall
from .avoid_obstacle import AvoidObstacle
from .find_color import FindColor
from .push_color import PushColor

# Definir qué se exporta cuando se usa "from behaviours import *"
__all__ = [
    'Behaviour',
    'AvoidFall',
    'AvoidObstacle',
    'FindColor',
    'PushColor'
]