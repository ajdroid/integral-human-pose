#Suhit Kodgule 04/26/19
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import ipdb

from .kin_layer_utils import *



class KinematicLayer(nn.Module):
    def __init__(self):
        super().__init__()

        #Set up the layer
        self.SHAPE_NORM = 300.0

        self.shape = torch.empty(Shape.NUM_SHAPE_PARAMETERS, 1).cuda()
        self.shape[Shape.Knee2Ankle] = 300.0
        self.shape[Shape.Hip2Knee] = 350.0
        self.shape[Shape.Pelvis2Hip] = 75.0
        self.shape[Shape.Pelvis2Torso] = 400.0
        self.shape[Shape.Torso2Neck] = 73.96
        self.shape[Shape.Neck2Head] = 249.03
        self.shape[Shape.Elbow2Wrist] = 250.0
        self.shape[Shape.Shoulder2Elbow] = 250.0
        self.shape[Shape.Torso2Shoulder] = 170.0
        self.shape = self.shape / self.SHAPE_NORM
        self.kinematic_operator = Operation(self.SHAPE_NORM)

        self.joint = [ [] for i in range(Joint.ORIG_NUM_JOINTS)]

        #Root
        self.joint[Joint.Pelvis].append(('global_trans', -1))
        self.joint[Joint.Pelvis].append(('rot_z', Motion.RotZ))
        self.joint[Joint.Pelvis].append(('rot_x', Motion.RotX))
        self.joint[Joint.Pelvis].append(('rot_y', Motion.RotY))

        #UpperBody
        self.joint[Joint.Torso] = self.joint[Joint.Pelvis].copy()
        self.joint[Joint.Torso].append(('rot_z', Motion.Torso_RotZ))
        self.joint[Joint.Torso].append(('rot_y', Motion.Torso_RotY))
        self.joint[Joint.Torso].append(('plus_y', Shape.Pelvis2Torso))
        
        self.joint[Joint.Neck] = self.joint[Joint.Torso].copy()
        self.joint[Joint.Neck].append(('rot_z', Motion.Neck_RotZ))
        self.joint[Joint.Neck].append(('rot_x', Motion.Neck_RotX))
        self.joint[Joint.Neck].append(('rot_y', Motion.Neck_RotY))
        self.joint[Joint.Neck].append(('plus_y', Shape.Torso2Neck))
    
        self.joint[Joint.Head] = self.joint[Joint.Neck].copy()
        self.joint[Joint.Head].append(('rot_x', Motion.Head_RotX))
        self.joint[Joint.Head].append(('plus_y', Shape.Neck2Head))
    
        #L_Leg
        self.joint[Joint.L_Hip] = self.joint[Joint.Pelvis].copy()
        self.joint[Joint.L_Hip].append(('plus_x', Shape.Pelvis2Hip))
    
        self.joint[Joint.L_Knee] = self.joint[Joint.L_Hip].copy()
        self.joint[Joint.L_Knee].append(('rot_z', Motion.L_Hip_RotZ))
        self.joint[Joint.L_Knee].append(('rot_x', Motion.L_Hip_RotX))
        self.joint[Joint.L_Knee].append(('rot_y', Motion.L_Hip_RotY))
        self.joint[Joint.L_Knee].append(('minus_y', Shape.Hip2Knee))
        
        self.joint[Joint.L_Ankle] = self.joint[Joint.L_Knee].copy()
        self.joint[Joint.L_Ankle].append(('rot_x', Motion.L_Knee_RotX))
        self.joint[Joint.L_Ankle].append(('minus_y', Shape.Knee2Ankle))
    
        #R_Leg
        self.joint[Joint.R_Hip] = self.joint[Joint.Pelvis].copy()
        self.joint[Joint.R_Hip].append(('minus_x', Shape.Pelvis2Hip))
        self.joint[Joint.R_Knee] = self.joint[Joint.R_Hip].copy()
        self.joint[Joint.R_Knee].append(('rot_z', Motion.R_Hip_RotZ))
        self.joint[Joint.R_Knee].append(('rot_x', Motion.R_Hip_RotX))
        self.joint[Joint.R_Knee].append(('rot_y', Motion.R_Hip_RotY))
        self.joint[Joint.R_Knee].append(('minus_y', Shape.Hip2Knee))
        
        self.joint[Joint.R_Ankle] = self.joint[Joint.R_Knee].copy()
        self.joint[Joint.R_Ankle].append(('rot_x', Motion.R_Knee_RotX))
        self.joint[Joint.R_Ankle].append(('minus_y', Shape.Knee2Ankle))
        
        #L_Arm
        self.joint[Joint.L_Shoulder] = self.joint[Joint.Torso].copy()
        self.joint[Joint.L_Shoulder].append(('plus_x', Shape.Torso2Shoulder))
    
        self.joint[Joint.L_Elbow] = self.joint[Joint.L_Shoulder].copy()
        self.joint[Joint.L_Elbow].append(('rot_z', Motion.L_Shoulder_RotZ))
        self.joint[Joint.L_Elbow].append(('rot_x', Motion.L_Shoulder_RotX))
        self.joint[Joint.L_Elbow].append(('rot_y', Motion.L_Shoulder_RotY))
        self.joint[Joint.L_Elbow].append(('minus_y', Shape.Shoulder2Elbow))
        
        self.joint[Joint.L_Wrist] = self.joint[Joint.L_Elbow].copy()
        self.joint[Joint.L_Wrist].append(('rot_x', Motion.L_Elbow_RotX))
        self.joint[Joint.L_Wrist].append(('minus_y', Shape.Elbow2Wrist))
    
        #R_Arm
        self.joint[Joint.R_Shoulder] = self.joint[Joint.Torso].copy()
        self.joint[Joint.R_Shoulder].append(('minus_x', Shape.Torso2Shoulder))
    
        self.joint[Joint.R_Elbow] = self.joint[Joint.R_Shoulder].copy()
        self.joint[Joint.R_Elbow].append(('rot_z', Motion.R_Shoulder_RotZ))
        self.joint[Joint.R_Elbow].append(('rot_x', Motion.R_Shoulder_RotX))
        self.joint[Joint.R_Elbow].append(('rot_y', Motion.R_Shoulder_RotY))
        self.joint[Joint.R_Elbow].append(('minus_y', Shape.Shoulder2Elbow))
        
        self.joint[Joint.R_Wrist] = self.joint[Joint.R_Elbow].copy()
        self.joint[Joint.R_Wrist].append(('rot_x', Motion.R_Elbow_RotX))
        self.joint[Joint.R_Wrist].append(('minus_y', Shape.Elbow2Wrist))

        related = [[False]*Joint.ORIG_NUM_JOINTS for _ in range(Motion.NUM_PARAMETERS)]

        for j in range(Motion.NUM_PARAMETERS):
            for i in range(Joint.ORIG_NUM_JOINTS):
                related[j][i] = False
                for k in range(len(self.joint[i]) -1, 0, -1):
                    if (self.joint[i][k][0] == 'rot_x') or (self.joint[i][k][0] == 'rot_y') \
                            or (self.joint[i][k][0] == 'rot_x') and (self.joint[i][k][1] == j):
                                related[j][i] = True
     

    def update(self, method_name, bottom_data, ind, v, scale=None):
        """
        Applies kinematic transformation to vector v
        params:
            method_name: string, name of function
            bottom_data: torch.Tensor, input motion_params
            ind: int, index of motion_param
            v: Torch.Tensor, input vector
            scale: int, some scale parameter for bone 
        return: Torch.Tensor, output vector 
        """
        ko = self.kinematic_operator(method_name,
                                        bottom_data, 
                                        ind, 
                                        scale, 
                                        self.shape)
        return ko.mm(v)

    def forward(self, x):
        bs = x.shape[0]
        # iterate through batch
        out = torch.empty((bs, 3*Joint.ORIG_NUM_JOINTS + 3)).cuda()
        for t in range(bs):
            # out[t].requires_grad = True
            scale = x[t, -1]
            motion_params = x[t, :-1]
            for i in range(Joint.ORIG_NUM_JOINTS):
                v = torch.Tensor([[0.], [0.], [0.], [1.]]).cuda()
                v.requires_grad = True
                # v = torch.transpose(v, 0)
                for k in range(len(self.joint[i])-1, 0, -1):
                    # import ipdb; ipdb.set_trace()
                    v = self.update(self.joint[i][k][0], motion_params, self.joint[i][k][1], v, scale)
                v = v.view(1,-1)
                out[t][3*i: 3*i + 3] = v[0,:-1].squeeze()

                out[t][-3:] = (out[t][8*3: 8*3 + 3] + out[t][6*3:6*3 + 3] )/ 2.

        return out
        
if __name__ == '__main__':
    KL = KinematicLayer()

    #Plus one for scale
    bottom_data = torch.zeros(Joint.ORIG_NUM_JOINTS * Motion.NUM_PARAMETERS + 1)
    # bottom_data[0] = 3.14
    bottom_data[15] = -3.14/4
    bottom_data.requires_grad = True
    bottom_data[-1] = 1
    # ind = 0
    # v = torch.ones(4,1)
    # v.requires_grad = True
    # scale = 1
    out= KL.forward(bottom_data)
    out_np = out.cpu().detach().numpy()
    # ipdb.set_trace()

    #Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = out_np[0::3]
    y = out_np[1::3]
    z = out_np[2::3]
    ax.scatter(out_np[0::3], out_np[1::3], out_np[2::3])
    ax.plot(out_np[0::3], out_np[1::3], out_np[2::3])

    plt.show()
    # v=KL.update('plus_y', bottom_data, ind, v, scale)
