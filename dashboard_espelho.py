#!/usr/bin/env python3
"""
Dashboard Espelho
- UI idêntica ao dashboard_integrado
- modo somente leitura (nunca executa build_core_analysis)
"""

import os

# Sinaliza modo espelho para o dashboard_integrado
os.environ["KINTUADI_DASHBOARD_MODE"] = "espelho"

from dashboard_integrado import main


if __name__ == "__main__":
    main()
