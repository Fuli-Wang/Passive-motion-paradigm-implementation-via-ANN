# ANN-based-approximate-kinematic-transformations

The "pmptrain.py" file is using ANN to fit the relationship between joint angles and end positions, evaluating of the Jacobian matrix via  tf.GradientTape API.


The "pmptrain_chain_rule.py" file is an example using the chain-rule to calculate the Jacobian matrix (Not recommanded for networks with a high number of layers and nerves).


The "IndhRobot_example.txt" and "OutdhRobot_example.txt" show the format of the training data：

"IndhRobot_example.txt" -- Joint angles (degree)

"OutdhRobot_example.txt" -- End position (mm)

The rows in both files should correspond to each other, where the joint angles correspond to their end positions.


Once the training is done, the model will be saved. To transfer or fine-tune the model via transfer learning (make sure the data and model loaded correctly):

    python3 transfertrain.py
