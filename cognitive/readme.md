## Three cognitive primitive evaluations

### Before or behind
Use a transparent glass and a goal. Place the goal behind the glass or before the glass. Classify.
Let agent take five steps for the message to propagate. Freeze and do DIR.

### Occulusion
Move object behind the occluding wall.
Classify at different frames whether the object is occluded.
Strapped observation.

### Rotation permanence
Find object.
Rotate 180.
Take four more actions: set the last few features as positive example.

For negative example, set the object to be invisible the entire time. Also take the last few features.

## General methodology
To create the evaluations to train MLP from, we
recognize a few constraints.

1) The environment yml needs to be procedurally generated to ensure diversity.

2) Data points are collected from running of a fixed agent.
   The agent's internal representations need to be retrieved to generate
   the dataset.
   
3) Visibility is achieved as the environment is defined. It has to be.