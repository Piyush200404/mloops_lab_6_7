#!/usr/bin/env bash
# Usage: bash scripts/sample_requests.sh [BASE_URL]
BASE=${1:-http://127.0.0.1:8000}
H='Content-Type: application/json'

echo "== health ==";             curl -s $BASE/health; echo
echo "== valid ==";              curl -s -X POST $BASE/predict -H "$H" -d '{"area_sqft":1800,"bedrooms":3,"bathrooms":2,"age_years":12,"location_score":7.5,"neighborhood":"suburban"}'; echo
echo "== missing field ==";      curl -s -X POST $BASE/predict -H "$H" -d '{"bedrooms":3,"bathrooms":2,"age_years":12,"location_score":7.5,"neighborhood":"suburban"}'; echo
echo "== wrong type ==";         curl -s -X POST $BASE/predict -H "$H" -d '{"area_sqft":"big","bedrooms":3,"bathrooms":2,"age_years":12,"location_score":7.5,"neighborhood":"suburban"}'; echo
echo "== invalid business ==";   curl -s -X POST $BASE/predict -H "$H" -d '{"area_sqft":-100,"bedrooms":3,"bathrooms":2,"age_years":12,"location_score":7.5,"neighborhood":"suburban"}'; echo
