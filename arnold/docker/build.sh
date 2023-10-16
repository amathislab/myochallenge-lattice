#!/bin/bash
set -e
set -u
SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/.."

docker build --network host -t myochallengeeval_mani_agent -f docker/Dockerfile .
# docker build --network host -t myo2023p1 -f docker/Dockerfile .
# evalai push myochallengeeval_loco_agent:latest --phase myochallenge2023-locophase1-2105
