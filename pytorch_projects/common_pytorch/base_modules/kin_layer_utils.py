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
                                'Neck2Head', 
                                'Elbow2Wrist', 
                                'Shoulder2Elbow', 
                                'Torso2Shoulder', 
                                'NUM_SHAPE_PARAMETERS'], count()))

Motion = IntEnum('Motion', zip(['TransX',
                                    'TransY',
                                    'TransZ',
                                    'RotZ',
                                    'RotX',
                                    'RotY',
                                    'Torso_RotY',
                                    'Torso_RotZ',
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
                                    'Neck_RotZ',
                                    'Neck_RotX',
                                    'Neck_RotY',
                                    'Head_RotX',
                                    'NUM_PARAMETERS'], count()))

Joint = IntEnum('Joint', zip(['R_Ankle',
                                'R_Knee',
                                'R_Hip',
                                'L_Hip',
                                'L_Knee',
                                'L_Ankle',
                                'Pelvis',
                                'Torso',
                                'Neck',
                                'Head',
                                'R_Wrist',
                                'R_Elbow',
                                'R_Shoulder',
                                'L_Shoulder',
                                'L_Elbow',
                                'L_Wrist',
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

    def __call__(self, method_name, bottom_data, ind, scale, shape):
        method = getattr(self, method_name, lambda: torch.ones(4,4))
        return method(bottom_data, ind, scale, shape)

    def rot_x(self, bottom_data, ind, scale, shape):
        M = torch.eye(4).cuda()
        M[1,1] = torch.cos(bottom_data[ind])
        M[1,2] = -torch.sin(bottom_data[ind])
        M[2,1] = torch.sin(bottom_data[ind])
        M[2,2] = torch.cos(bottom_data[ind])
        return M

    def rot_y(self, bottom_data, ind, scale, shape):
        M = torch.eye(4).cuda()
        M[0,0] = torch.cos(bottom_data[ind])
        M[0,2] = torch.sin(bottom_data[ind])
        M[2,0] = -torch.sin(bottom_data[ind])
        M[2,2] = torch.cos(bottom_data[ind])
        return M

    def rot_z(self, bottom_data, ind, scale, shape):
        M = torch.eye(4).cuda()
        M[0,0] = torch.cos(bottom_data[ind])
        M[0,1] = -torch.sin(bottom_data[ind])
        M[1,0] = torch.sin(bottom_data[ind])
        M[1,1] = torch.cos(bottom_data[ind])
        return M

    def global_trans(self, bottom_data, ind, scale, shape):
        M = torch.eye(4).cuda()
        M[0,3] = bottom_data[Motion.TransX]
        M[1,3] = bottom_data[Motion.TransY]
        M[2,3] = bottom_data[Motion.TransZ]
        return M
        
    def plus_x(self, bottom_data, ind, scale, shape):
#         if scale == None:
#             raise ValueError('Scale not passed')
  #       if shape == None:
  #           raise ValueError('Shape not passed')
        M = torch.eye(4).cuda()
        M[0,3] = shape[ind] * scale * self.SHAPE_NORM
        return M

    def plus_y(self, bottom_data, ind, scale, shape):
   #      if scale == None:
   #          raise ValueError('Scale not passed')
        # if shape == None:
        #     raise ValueError('Shape not passed')
        M = torch.eye(4).cuda()
        M[1,3] = shape[ind] * scale * self.SHAPE_NORM
        return M

    def minus_x(self, bottom_data, ind, scale, shape):
     #    if scale == None:
     #        raise ValueError('Scale not passed')
        # if shape == None:
        #     raise ValueError('Shape not passed')
        M = torch.eye(4).cuda()
        M[0,3] = -shape[ind] * scale * self.SHAPE_NORM
        return M

    def minus_y(self, bottom_data, ind, scale, shape):
       #  if scale == None:
       #      raise ValueError('Scale not passed')
#         if shape == None:
#             raise ValueError('Shape not passed')
        M = torch.eye(4).cuda()
        # import ipdb; ipdb.set_trace()
        M[1,3] = -shape[ind] * scale * self.SHAPE_NORM
        return M


