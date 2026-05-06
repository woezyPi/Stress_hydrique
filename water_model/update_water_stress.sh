#!/bin/bash
# ============================================================
# GreenDC Water Stress — Mise à jour mensuelle automatique
#
# Cron (1er de chaque mois à 3h) :
#   0 3 1 * * /home/alexis/Greendc/water_model/update_water_stress.sh >> /var/log/greendc_water.log 2>&1
#
# ERA5 a un délai de ~3 mois → en janvier on a les données d'octobre
# ============================================================

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$REPO_DIR/venv/bin/activate"
LOG_PREFIX="[$(date '+%Y-%m-%d %H:%M:%S')] [water-stress]"

echo "$LOG_PREFIX Début mise à jour"

# Activer le venv
source "$VENV"

# Année cible = année courante - 1 si on est avant octobre, sinon année courante
# (ERA5 publie l'année N complète vers mars N+1)
CURRENT_MONTH=$(date +%m)
CURRENT_YEAR=$(date +%Y)
if [ "$CURRENT_MONTH" -ge 4 ]; then
    TARGET_YEAR=$((CURRENT_YEAR - 1))
else
    TARGET_YEAR=$((CURRENT_YEAR - 2))
fi

echo "$LOG_PREFIX Calcul pour l'année $TARGET_YEAR"

# Lancer le pipeline (télécharge ERA5 si absent, sinon utilise le cache)
cd "$REPO_DIR"
python -m water_model.pipeline --year "$TARGET_YEAR"

echo "$LOG_PREFIX Terminé. Sorties : water_model/data/outputs/"
