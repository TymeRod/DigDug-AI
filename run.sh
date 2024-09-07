#!/bin/bash

# Define the usage function
usage() {
    echo "Usage: $0 [-f <file>] [-d <directory>] [-h]"
    echo "Options:"
    echo "  -p <port>       Port number(s) (comma-separated)"
    echo "  -r <amount>     Number of runs"
    echo "  -h              Display this help message"
    exit 1
}

no_arg=true
runs=1

# Parse the command line arguments
while getopts ":p:r:h" opt; do
    case $opt in
        p)
            ports=$OPTARG
            ;;
        r)
            runs=$OPTARG
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
    no_arg=false
done

# Check if no argument is provided
if [ "$no_arg" = true ]; then
    gnome-terminal --tab --title="server" -- python3 server.py &
    gnome-terminal --tab --title="viewer" -- python3 viewer.py &
    gnome-terminal --tab --title="main" -- python3 main.py  &
    wait $!
fi

# Check if the only argument passed is -r
if [ "$no_arg" = false ] && [ "$runs" != 1 ] && [ -z "$ports" ]; then
    gnome-terminal --tab --title="server" -- python3 server.py &
    gnome-terminal --tab --title="viewer" -- python3 viewer.py &
    for i in $(seq 1 $runs); do
        gnome-terminal --tab --title="main_run_$i" -- python3 main.py &
        wait $!
    done
fi

# Add your code here to perform actions based on the provided options
# Check if the ports option is provided
if [ -n "$ports" ]; then
    IFS=',' read -ra port_array <<< "$ports"
    counter=1
    pids=()
    for port in "${port_array[@]}"; do
        gnome-terminal --tab --title="server_$counter" -- python3 server.py --port $port &
        gnome-terminal --tab --title="viewer_$counter" -- python3 viewer.py --port $port &
        counter=$((counter+1))
    done

    for i in $(seq 1 $runs); do
        counter=1
        for port in "${port_array[@]}"; do
            gnome-terminal --tab --title="main_${counter}_run_$i" -- python3 main.py --port $port &
            pids+=($!)
            counter=$((counter+1))
        done

        for pid in "${pids[@]}"; do
            wait $pid
        done
    done

fi
