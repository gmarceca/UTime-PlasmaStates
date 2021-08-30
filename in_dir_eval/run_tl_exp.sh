#!/bin/bash

ut train_plasma_states_detector --num_GPUs=1 --n_epochs=25 --fold='1' --InverseGradualTraining=true
wait
ut train_plasma_states_detector --num_GPUs=1 --n_epochs=50 --fold='1' --continue_training
wait
ut train_plasma_states_detector --num_GPUs=1 --n_epochs=25 --fold='2' --InverseGradualTraining=true
wait
ut train_plasma_states_detector --num_GPUs=1 --n_epochs=50 --fold='2' --continue_training
wait
ut train_plasma_states_detector --num_GPUs=1 --n_epochs=25 --fold='3' --InverseGradualTraining=true
wait
ut train_plasma_states_detector --num_GPUs=1 --n_epochs=50 --fold='3' --continue_training
wait
ut train_plasma_states_detector --num_GPUs=1 --n_epochs=50 --fold='4'
wait
ut train_plasma_states_detector --num_GPUs=1 --n_epochs=50 --fold='5'
wait
ut train_plasma_states_detector --num_GPUs=1 --n_epochs=50 --fold='6'
