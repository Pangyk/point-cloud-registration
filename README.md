Python 3.7

# Introduction:
- The colon dir contans code to simulate colons (I only changed some parameters and added some downsampling method). 
- The reg dir contains code to register colons. The method takes .obj file as input.

# Steps:
- First perform rigid registration use "python /robot_curve/reg/cpd_reg.py"
- Then use "python /robot_curve/reg/robot/demos/toy_reg.py" to start deformable registration.
- Finally use "python /robot_curve/reg/evaluate.py" to evaluate the performance. 


# Requirements
- In /robot_curve/reg/robot/requirement.txt
- Please manually install a pytorch >= 1.5 (but < 2.0.0) with CUDA toolkit < 11. (Otherwise the pykeops will stuck)
- Open3D 0.13.0 might have problem opening .obj file (wrong number of points). Please pip install open3d==0.9.0 or use versions higher than 0.13.0 (not sure which version is safe)



