#!/bin/bash

WORKERS=('api2app-workers/whisper-cpp.py')
WORKERS_NUM=(1)
QUEUE_SIZE_URLS=('https://queue.api2app.org/queue_size/ea474c9a-c631-4bac-9141-ba26b0ff56c5')

DIR="$(pwd)"
ACTION="none"

GREEN="\e[32m"
RED="\e[31m"
GRAY="\e[2m"
BLUE="\e[94m\e[1m"
NC="\e[0m"
BOLD="\e[1m"

source "${DIR}/venv/bin/activate"

if [ "$1" == "-h" ]; then
    echo -e "$NC"
    echo -e "${BLUE}Usage: ./$(basename "$0") status|start|stop"
    echo -e "$NC"
    exit 0
fi

if [ -n "$1" ]; then
    ACTION="$1"
else
    exit 0
fi

function count_items() {
    local search_item=$1
    shift
    local array=("$@")
    local COUNT=0
    for i in "${array[@]}"; do
        if [ "$i" == "$search_item" ]; then
            COUNT=$(expr $COUNT + 1)
        fi
    done
    echo "$COUNT"
}

PIDS="$(pidof python)"
echo -e "${BLUE}PIDs: ${PIDS}""$NC"
IFS=' '
read -ra PIDS_ARR <<< "$PIDS"

PS_ARR=()

for PID in "${PIDS_ARR[@]}"; do
    PS_OUT=$(ps "$PID" | grep 'python' --color=none)
    read -ra PS_OUT_ARR <<< "$PS_OUT"
    PS_ARR=("${PS_ARR[@]}" "${PS_OUT_ARR[5]}")
done

if [ $ACTION == 'status' ]; then
    echo -e "$NC"
    echo -e "${GRAY}-----------------------------------------"
    echo "                  STATUS                  "
    echo "-----------------------------------------"
    echo -e "$NC"

    for index in "${!WORKERS[@]}"; do
    	  worker_name="${WORKERS[$index]}"
        count=$(count_items "$worker_name" "${PS_ARR[@]}")
        if [[ "$count" -lt "${WORKERS_NUM[$index]}" ]]; then
            echo -e "${RED}- ${worker_name} [${count}]${NC}"
        else
            echo -e "${GREEN}- ${worker_name} [${count}]${NC}"
        fi
    done

    for queue_size_url in "${QUEUE_SIZE_URLS[@]}"; do
        # echo -e "${GRAY}Queue URL: ${queue_size_url}"
        queue_size=$(curl -s "$queue_size_url" | python3 -c "import sys, json; print(json.load(sys.stdin)['queue_size'])")
        echo -e "${GRAY}Queue size: ${queue_size}"
    done
    echo -e "$NC"
fi

if [ $ACTION == 'start' ]; then
    echo -e "$NC"
    echo -e "${GRAY}-----------------------------------------"
    echo "                  START                  "
    echo "-----------------------------------------"
    echo -e "$NC"

    STARTED_COUNT=0
    i=0
    for index in "${!WORKERS[@]}"; do
        worker_name="${WORKERS[$index]}"
        count=$(count_items "$worker_name" "${PS_ARR[@]}")
        NUM=$((WORKERS_NUM[$index] - count))
        if [[ "$count" -lt "${WORKERS_NUM[$index]}" ]]; then

            for (( j=0; j<${NUM}; j++ )); do
                echo -e "${GRAY}${BOLD}Starting${NC} ${worker_name}"
                nohup python "${worker_name}" > "${worker_name/\//_}"_log.txt 2>&1 &
                ((STARTED_COUNT++))
            done

        fi
        ((i++))
    done

    echo -e "${GREEN}Started: ${STARTED_COUNT}"
fi

if [ $ACTION == 'stop' ]; then
    echo -e "$NC"
    echo -e "${GRAY}-----------------------------------------"
    echo "                  STOP                  "
    echo "-----------------------------------------"
    echo -e "$NC"

    STOPPED_COUNT=0

    for PID in "${PIDS_ARR[@]}"; do
        PS_OUT=$(ps "$PID" | grep 'python' --color=none)
        read -ra PS_OUT_ARR <<< "$PS_OUT"
        for worker_name in "${WORKERS[@]}"; do
            if [ "$worker_name" == "${PS_OUT_ARR[5]}" ]; then
                echo -e "${GRAY}${BOLD}Stopping ${PID}${NC}" "${PS_OUT_ARR[5]}"
                kill "$PID"
                ((STOPPED_COUNT++))
            fi
        done
    done

    echo -e "${GREEN}Stopped: ${STOPPED_COUNT}"
fi
