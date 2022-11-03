# Первый аргумент скрипта - путь до аналога grid.flows.xml
FLOWS_PATH="${1:-grid.flows.xml}"
jtrrouter -n grid.net.xml \
    --remove-loops \
    -T 25,25,25,25 \
    --sink-edges B1B2,C1C2,C1D1 \
    --route-files $FLOWS_PATH \
    -o tmp/grid.routes.xml
