"""Garante que a raiz do projeto esteja no sys.path durante os testes.

Sem isto, `import planta_baixa` e `import plantabaixa` dependem de onde o pytest
foi chamado.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
