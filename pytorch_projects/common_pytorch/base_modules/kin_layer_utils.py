#Suhit Kodgule 04/26/19
from itertools import count
from enum import IntEnum

import torch

# Operation = IntEnum('Operation', zip(['rot_x', 
                                        # 'rot_y', 
                                        # 'rot_z', 
                                        # 'global_trans', 
                                        # 'plus_x',
                                        # 'plus_y', 
                                        # 'minus_x', 
#                                         'minus_y'], count()))

#Set up enumerators for easy indexing

Shape = IntEnum('Shape', zip(['Knee2Ankle', 
                                'Hip2Knee', 
                                'Pelvis2Hip', 
                                'Pelvis2Torso',
                                'Torso2Neck', 
                                'Nose2HeadY',
                                'Elbow2Wrist', 
                                'Shoulder2Elbow', 
                                'Neck2ShoulderX',
                                'Neck2ShoulderY',
                                'Neck2NoseY',
                                'Neck2NoseZ',
                                'NUM_SHAPE_PARAMETERS'], count()))

Motion = IntEnum('Motion', zip(['TransX',
                                    'TransY',
                                    'TransZ',
                                    'RotZ',
                                    'RotX',
                                    'RotY',
                                    'Pelvis_RotY',
                                    'Pelvis_RotX',
                                    'Pelvis_RotZ',
                                    'L_Hip_RotZ',
                                    'L_Hip_RotX',
                                    'L_Hip_RotY',
                                    'L_Knee_RotX',
                                    'R_Hip_RotZ',
                                    'R_Hip_RotX',
                                    'R_Hip_RotY',
                                    'R_Knee_RotX',
                                    'L_Shoulder_RotZ',
                                    'L_Shoulder_RotX',
                                    'L_Shoulder_RotY',
                                    'L_Elbow_RotX',
                                    'R_Shoulder_RotZ',
                                    'R_Shoulder_RotX',
                                    'R_Shoulder_RotY',
                                    'R_Elbow_RotX',
                                    'Neck_RotZ_H',
                                    'Neck_RotX_H',
                                    'Neck_RotY_H',
                                    'Head_RotX',
                                    'NUM_PARAMETERS'], count()))

Joint = IntEnum('Joint', zip(['Pelvis',
                                'R_Hip',
                                'R_Knee',
                                'R_Ankle',
                                'L_Hip',
                                'L_Knee',
                                'L_Ankle',
                                'Torso',
                                'Neck',
                                'Nose',
                                'Head',
                                'L_Shoulder',
                                'L_Elbow',
                                'L_Wrist',
                                'R_Shoulder',
                                'R_Elbow',
                                'R_Wrist',
                                'Thorax',
                                'ORIG_NUM_JOINTS'], count()))

#Define Kinematic Operator
class Operation(object):
    """
    Transformation Operation class.
    Initialization params: None
    Callback params:
        method_name: string, name of function
        bottom_data: torch.Tensor, input motion_params
        ind: IntEnum.element, index of motion_param
        scale: int, some scale parameter for bone 
        shape: torch.Tensor, bone length vector
    return: torch.Tensor, Transformation Matrix M
    """
    def __init__(self,SHAPE_NORM):
        self.SHAPE_NORM = SHAPE_NORM
        return

    def __call__(self, method_name, bottom_data, ind, scale, shape, M):
        method = getattr(self, method_name, lambda: None)
        return method(bottom_data, ind, scale, shape, M)

    def rot_x(self, bottom_data, ind, scale, shape, M):
        M[1,1] = torch.cos(bottom_data[ind])
        M[1,2] = -torch.sin(bottom_data[ind])
        M[2,1] = torch.sin(bottom_data[ind])
        M[2,2] = torch.cos(bottom_data[ind])
        return M

    def rot_y(self, bottom_data, ind, scale, shape, M):
        M[0,0] = torch.cos(bottom_data[ind])
        M[0,2] = torch.sin(bottom_data[ind])
        M[2,0] = -torch.sin(bottom_data[ind])
        M[2,2] = torch.cos(bottom_data[ind])
        return M

    def rot_z(self, bottom_data, ind, scale, shape, M):
        M[0,0] = torch.cos(bottom_data[ind])
        M[0,1] = -torch.sin(bottom_data[ind])
        M[1,0] = torch.sin(bottom_data[ind])
        M[1,1] = torch.cos(bottom_data[ind])
        return M

    def global_trans(self, bottom_data, ind, scale, shape, M):
        M[0,3] = bottom_data[Motion.TransX]
        M[1,3] = bottom_data[Motion.TransY]
        M[2,3] = bottom_data[Motion.TransZ]
        return M
        
    def plus_x(self, bottom_data, ind, scale, shape, M):
        M[0,3] = shape[ind] * scale * self.SHAPE_NORM
        return M

    def plus_y(self, bottom_data, ind, scale, shape, M):
        M[1,3] = shape[ind] * scale * self.SHAPE_NORM
        return M

    def plus_z(self, bottom_data, ind, scale, shape, M):
        M[2, 3] = shape[ind] * scale * self.SHAPE_NORM
        return M

    def minus_x(self, bottom_data, ind, scale, shape, M):
        M[0,3] = -shape[ind] * scale * self.SHAPE_NORM
        return M

    def minus_y(self, bottom_data, ind, scale, shape, M):
        M[1,3] = -shape[ind] * scale * self.SHAPE_NORM
        return M

    def minus_z(self, bottom_data, ind, scale, shape, M):
        M[2, 3] = -shape[ind] * scale * self.SHAPE_NORM
        return M

