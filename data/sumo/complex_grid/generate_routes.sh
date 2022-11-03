# Первый аргумент скрипта - путь до аналога grid.flows.xml
FLOWS_PATH="${1:-grid.flows.xml}"
jtrrouter -n grid.net.xml \
    --remove-loops \
    -T 25,25,25,25 \
    --sink-edges B2A2,B1A1,B1B0,C1C0,C1D1,C2D2,C2C3,B2B3 \
    --route-files $FLOWS_PATH \
    -o tmp/grid.routes.xml
