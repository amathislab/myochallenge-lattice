#!/bin/bash
set -e
set -u

if [ $# -eq 0 ]
then
    echo "running docker without display"
    docker run -it --network=host --gpus=all myochallengeeval_mani_agent python /home/gymuser/src/main_challenge_manipulation_phase2.py \
	--use_lattice --load_path="/home/gymuser/output/trained_agents/curriculum_step_9"
else
    export DISPLAY=$DISPLAY
	echo "setting display to $DISPLAY"
	xhost +
	docker run -it --rm -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network=host --gpus=all myochallengeeval_mani_agent python /home/gymuser/src/main_challenge_manipulation_phase2.py \
	--use_lattice --load_path="/home/gymuser/output/trained_agents/curriculum_step_9"
	xhost -
fi
