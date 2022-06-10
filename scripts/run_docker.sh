#!/bin/bash
# Launch an experiment using the docker cpu image
# Only works by default for mac and linux

JULIA_CMD=julia
DOCKER_IMAGE=iai
SYSIMAGE_FILE=iai-config/sys.so

export IAI_LICENSE_FILE=iai-config/iai-server.lic
nohup "$JULIA_CMD" --sysimage="$SYSIMAGE_FILE" -e "IAI.IAILicensing.start_server(); wait(Condition())" > /dev/null 2>&1 &
SERVER_PID=$!

CMD_LINE="$@"

echo "Executing in the docker (cpu image):"
echo $CMD_LINE

docker run -it --network host --rm --ipc=host -p 8888:8888 \
 --mount src=$(pwd),target=/root/code/rl_zoo,type=bind $DOCKER_IMAGE\
  bash -c "cd /root/code/rl_zoo/ && $CMD_LINE"

kill $SERVER_PID