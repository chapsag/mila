#!/bin/bash

echo 'setup env for python this may take a while... 🧙'


progress-bar() {
  local duration=${1}


    already_done() { for ((done=0; done<$elapsed; done++)); do printf "▇"; done }
    remaining() { for ((remain=$elapsed; remain<$duration; remain++)); do printf " "; done }
    percentage() { printf "| %s%%" $(( (($elapsed)*100)/($duration)*100/100 )); }
    clean_line() { printf "\r"; }

  for (( elapsed=1; elapsed<=$duration; elapsed++ )); do
      already_done; remaining; percentage
      sleep 0.01
      clean_line
  done
  clean_line
}



python3 -m venv .venv
source .venv/bin/activate
pip install numpy
pip install matplotlib

progress-bar 100
printf '\nFinished! Happy coding 🎉\n'