# CSGO-Player-Position-Prediction-with-RNN
Predicting players position in demo (and possibly in game) with RNN and LSTM

This repo predicts the position of the players one side at a time.

Huge shoutout to [awpy](https://github.com/pnxenopoulos/awpy) for providing a way to parse CS:GO demos.  

# Todo
- [ ] Experiment with more timestep
- [ ] Change hardcoded image length and width.
- [ ] Automate the entire process into a script

# How to use this repo
Download the script, download a demo that you wish to analyze, and download the layout image of the map in the demo.

For the script, you need to pass in these arguments:
- `--layout`: path to map layout image
- `--demo`: path to demo file
- `--side`: side of players to predict positions, values: (`ct` or `t`)
- `--model`: model type, values: (`rnn` or `ltsm`)
- `--output`: path of output predicted png files
- `--debug`: optional, no effect yet.
