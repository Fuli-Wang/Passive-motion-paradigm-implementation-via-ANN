# We observed that changing the Jacobian source (e.g., from an ANN-based estimate to a numerically computed Jacobian) while keeping all other parameters nominally “unchanged” can alter the effective closed-loop gain and lead to different behaviours. Now, we are currently redeveloping the entire framework, explicitly formalising and standardising the entire control law. The new implementation will be available as soon as possible.

# ANN-based-approximate-kinematic-transformations

The "pmptrain.py" file uses ANN to fit the relationship between joint angles and end positions, evaluating the Jacobian matrix via the TF GradientTape API.


The "pmptrain_chain_rule.py" file is an example of using the chain rule to calculate the Jacobian matrix (Not recommended for networks with a high number of layers and nerves).


The "IndhRobot_example.txt" and "OutdhRobot_example.txt" show the format of the training data：

"IndhRobot_example.txt" -- Joint angles (degree)

"OutdhRobot_example.txt" -- End position (mm)

The rows in both files should correspond to each other, where the joint angles correspond to their end positions.

To train an ANN model using the data:

    python3 pmptrain.py

There is another training file based on PyTorch (pmptrain_pytorch.py). The PyTorch-based PMP calculation process appears to be more efficient (it has not been fully verified). If you wish to use the PyTorch-based PMP, please use the pmptrain_pytorch.py file to train the model (or use keras2pth.py to transfer .keras file to .pth file) and update the pmp_ANN.py file accordingly to adapt the PyTorch. 

Once the training is done, the model will be saved. To transfer or fine-tune the model via transfer learning (make sure the data and model loaded correctly):

    python3 transfertrain.py

# Passive motion paradigm implementation via deep neural networks
We also provide a Python version of the Passive Motion Paradigm (PMP) motion model, which uses the deep neural networks-based PMP to realize the goal-directed motion:

    python3 pmp_ANN.py

## Citation
If you find this work useful, please cite our paper:

@article{Wang_Mohan_Tiwari_2025, title={Passive motion paradigm implementation via deep neural networks: analysis and verification}, volume={43}, DOI={10.1017/S0263574725000505}, number={5}, journal={Robotica}, author={Wang, Fuli and Mohan, Vishwanathan and Tiwari, Ashutosh}, year={2025}, pages={1766–1784}}
