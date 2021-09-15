# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
from typing import List
try:
    from box_intersection import batch_intersect
except ImportError:
    print("Could not import cythonized batch_intersection")
    batch_intersect = None


import numpy as np
from scipy.spatial import ConvexHull

def polygon_clip(subjectPolygon, clipPolygon):
   """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
    #   diff_cp = cp2 - cp1
    #   diff_p = p - cp1
    #   diff_p = diff_p[[1, 0]]
    #   mult = diff_cp * diff_p
    #   return mult[0] > mult[1]
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
    #   dc = cp1 - cp2
    #   dp = s - e
    #   n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    #   n2 = s[0] * e[1] - s[1] * e[0]
    #   n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    #   return (n1 * dp -  n2 * dc) * n3
   
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)


def helper_computeIntersection(cp1: torch.Tensor, cp2: torch.Tensor, s: torch.Tensor, e: torch.Tensor):
    dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
    dp = [ s[0] - e[0], s[1] - e[1] ]
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0] 
    n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
    # return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
    return torch.stack([(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3])

def helper_inside(cp1: torch.Tensor, cp2: torch.Tensor, p: torch.Tensor):
      ineq = (cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
      return ineq.item()

def polygon_clip_unnest(subjectPolygon: torch.Tensor, clipPolygon: torch.Tensor):
    """ Clip a polygon with another polygon.

   Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

   Args:
     subjectPolygon: a list of (x,y) 2d points, any polygon.
     clipPolygon: a list of (x,y) 2d points, has to be *convex*
   Note:
     **points have to be counter-clockwise ordered**

   Return:
     a list of (x,y) vertex point for the intersection polygon.
   """
    outputList = [subjectPolygon[x] for x in range(subjectPolygon.shape[0])]
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList.copy()
        outputList.clear()
        s = inputList[-1]
 
        for subjectVertex in inputList:
            e = subjectVertex
            if helper_inside(cp1, cp2, e):
                if not helper_inside(cp1, cp2, s):
                    outputList.append(helper_computeIntersection(cp1, cp2, s, e))
                outputList.append(e)
            elif helper_inside(cp1, cp2, s):
                outputList.append(helper_computeIntersection(cp1, cp2, s, e))
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            # return None
            break
    return outputList


def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def poly_area_tensor(x, y):
    return 0.5*torch.abs(torch.dot(x,torch.roll(y,1))-torch.dot(y,torch.roll(x,1)))

def box3d_vol_tensor(corners):
    EPS = 1e-6
    reshape = False
    B, K = corners.shape[0], corners.shape[1]
    if len(corners.shape) == 4:
        # batch x prop x 8 x 3
        reshape = True
        corners = corners.view(-1, 8, 3)
    a = torch.sqrt((corners[:, 0, :] - corners[:, 1, :]).pow(2).sum(dim=1).clamp(min=EPS))
    b = torch.sqrt((corners[:, 1, :] - corners[:, 2, :]).pow(2).sum(dim=1).clamp(min=EPS))
    c = torch.sqrt((corners[:, 0, :] - corners[:, 4, :]).pow(2).sum(dim=1).clamp(min=EPS))
    vols = a * b * c
    if reshape:
        vols = vols.view(B, K)
    return vols

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def enclosing_box3d_vol(corners1, corners2):
    """
    volume of enclosing axis-aligned box
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape)== 4
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners2.shape[2] == 8
    assert corners2.shape[3] == 3
    EPS = 1e-6

    corners1 = corners1.clone()
    corners2 = corners2.clone()
    # flip Y axis, since it is negative
    corners1[:, :, :, 1] *= -1
    corners2[:, :, :, 1] *= -1
    
    # min_a = torch.min(corners1[:, :, 0, :][:, :, None, :] , corners2[:, :, 0, :][:, None, :, :])
    # max_a = torch.max(corners1[:, :, 1, :][:, :, None, :] , corners2[:, :, 1, :][:, None, :, :])
    # a = (max_a - min_a).pow(2).sum(dim=3).clamp(min=EPS).sqrt()

    # min_b = torch.min(corners1[:, :, 1, :][:, :, None, :] , corners2[:, :, 1, :][:, None, :, :])
    # max_b = torch.max(corners1[:, :, 2, :][:, :, None, :] , corners2[:, :, 2, :][:, None, :, :])
    # b = (max_b - min_b).pow(2).sum(dim=3).clamp(min=EPS).sqrt()

    # min_c = torch.min(corners1[:, :, 0, :][:, :, None, :] , corners2[:, :, 0, :][:, None, :, :])
    # max_c = torch.max(corners1[:, :, 4, :][:, :, None, :] , corners2[:, :, 4, :][:, None, :, :])
    # c = (max_c - min_c).pow(2).sum(dim=3).clamp(min=EPS).sqrt()

    # vol = a * b * c
    
    al_xmin = torch.min( torch.min(corners1[:, :, :, 0], dim=2).values[:, :, None], torch.min(corners2[:, :, :, 0], dim=2).values[:, None, :])
    al_ymin = torch.max( torch.max(corners1[:, :, :, 1], dim=2).values[:, :, None], torch.max(corners2[:, :, :, 1], dim=2).values[:, None, :])
    al_zmin = torch.min( torch.min(corners1[:, :, :, 2], dim=2).values[:, :, None], torch.min(corners2[:, :, :, 2], dim=2).values[:, None, :])
    al_xmax = torch.max( torch.max(corners1[:, :, :, 0], dim=2).values[:, :, None], torch.max(corners2[:, :, :, 0], dim=2).values[:, None, :])
    al_ymax = torch.min( torch.min(corners1[:, :, :, 1], dim=2).values[:, :, None], torch.min(corners2[:, :, :, 1], dim=2).values[:, None, :])
    al_zmax = torch.max( torch.max(corners1[:, :, :, 2], dim=2).values[:, :, None], torch.max(corners2[:, :, :, 2], dim=2).values[:, None, :])

    diff_x = torch.abs(al_xmax - al_xmin)
    diff_y = torch.abs(al_ymax - al_ymin)
    diff_z = torch.abs(al_zmax - al_zmin)
    vol = diff_x * diff_y * diff_z
    return vol


def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.

    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU

    todo (rqi): add more description on corner points' orders.
    '''
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)]
    inter, inter_area = convex_hull_intersection(rect1, rect2)

    # corner points are in counter clockwise order
    # area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    # area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    
    # iou_2d = inter_area/(area1+area2-inter_area)

    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    union = (vol1 + vol2 - inter_vol)
    iou = inter_vol / union
    return iou, union


@torch.jit.ignore
def to_list_1d(arr) -> List[float]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr

@torch.jit.ignore
def to_list_3d(arr) -> List[List[List[float]]]:
    arr = arr.detach().cpu().numpy().tolist()
    return arr



def generalized_box3d_iou_tensor_non_diff(corners1: torch.Tensor, corners2: torch.Tensor, nums_k2: torch.Tensor, rotated_boxes: bool = True,
                                          return_inter_vols_only: bool = False,
                                          approximate: bool = True):
    if batch_intersect is None:
        return generalized_box3d_iou_tensor_jit(corners1, corners2, nums_k2, rotated_boxes, return_inter_vols_only)
    else:
        assert len(corners1.shape) == 4
        assert len(corners2.shape)== 4
        assert corners1.shape[2] == 8
        assert corners1.shape[3] == 3
        assert corners1.shape[0] == corners2.shape[0]
        assert corners1.shape[2] == corners2.shape[2]
        assert corners1.shape[3] == corners2.shape[3]

        B, K1 = corners1.shape[0], corners1.shape[1]
        _, K2 = corners2.shape[0], corners2.shape[1]

        # # box height. Y is negative, so max is torch.min
        ymax = torch.min(corners1[:, :, 0,1][:, :, None], corners2[:, :, 0,1][:, None, :])
        ymin = torch.max(corners1[:, :, 4,1][:, :, None], corners2[:, :, 4,1][:, None, :])
        height = (ymax - ymin).clamp(min=0)
        EPS = 1e-8
        
        idx = torch.arange(start=3, end=-1, step=-1, device=corners1.device)
        idx2 = torch.tensor([0,2], dtype=torch.int64, device=corners1.device)
        rect1 = corners1[:, :, idx, :]
        rect2 = corners2[:, :, idx, :]
        rect1 = rect1[:, :, :, idx2]
        rect2 = rect2[:, :, :, idx2]

        lt = torch.max(rect1[:, :, 1][:, :, None, :], rect2[:, :, 1][:, None, : ,:])
        rb = torch.min(rect1[:, :, 3][:, :, None, :], rect2[:, :, 3][:, None, : ,:])
        wh = (rb - lt).clamp(min=0)
        non_rot_inter_areas = wh[:, :, :, 0] * wh[:, :, :, 1]
        non_rot_inter_areas = non_rot_inter_areas.view(B, K1, K2)
        if nums_k2 is not None:
            for b in range(B):
                non_rot_inter_areas[b, :, nums_k2[b]:] = 0

        enclosing_vols = enclosing_box3d_vol(corners1, corners2)
        
        # vols of boxes
        vols1 = box3d_vol_tensor(corners1).clamp(min=EPS)
        vols2 = box3d_vol_tensor(corners2).clamp(min=EPS)
        
        sum_vols = vols1[:, :, None] + vols2[:, None, :]

        # filter malformed boxes
        good_boxes = (enclosing_vols > 2*EPS) * (sum_vols > 4*EPS)
        if rotated_boxes:
            inter_areas = np.zeros((B, K1, K2), dtype=np.float32)
            rect1 = rect1.cpu().detach().numpy()
            rect2 = rect2.cpu().detach().numpy()
            nums_k2_np = nums_k2.cpu().numpy()
            non_rot_inter_areas_np = non_rot_inter_areas.cpu().detach().numpy()
            batch_intersect(rect1, rect2, non_rot_inter_areas_np, nums_k2_np, inter_areas, approximate)
            inter_areas = torch.from_numpy(inter_areas)
        else:
            inter_areas = non_rot_inter_areas
       
        inter_areas = inter_areas.to(corners1.device)
        ### gIOU = iou - (1 - sum_vols/enclose_vol)
        inter_vols = inter_areas * height
        if return_inter_vols_only:
            return inter_vols

        union_vols = (sum_vols - inter_vols).clamp(min=EPS)
        ious = inter_vols / union_vols
        giou_second_term = - (1 - union_vols / enclosing_vols)
        gious = ious + giou_second_term
        gious *= good_boxes
        if nums_k2 is not None:
            mask = torch.zeros((B, K1, K2), device=height.device, dtype=torch.float32)
            for b in range(B):
                mask[b,:,:nums_k2[b]] = 1
            gious *= mask
        return gious



def generalized_box3d_iou_tensor(corners1: torch.Tensor, corners2: torch.Tensor, nums_k2: torch.Tensor, rotated_boxes: bool = True,
    return_inter_vols_only: bool = False, no_grad: bool = False):
    """
    Input:
        corners1: torch Tensor (B, K1, 8, 3), assume up direction is negative Y
        corners2: torch Tensor (B, K2, 8, 3), assume up direction is negative Y
        Assumes that the box is only rotated along Z direction
    Returns:
        B x K1 x K2 matrix of generalized IOU by approximating the boxes to be axis aligned
        The return IOU is differentiable
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape)== 4
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == corners2.shape[2]
    assert corners1.shape[3] == corners2.shape[3]

    B, K1 = corners1.shape[0], corners1.shape[1]
    _, K2 = corners2.shape[0], corners2.shape[1]

    # # box height. Y is negative, so max is torch.min
    ymax = torch.min(corners1[:, :, 0,1][:, :, None], corners2[:, :, 0,1][:, None, :])
    ymin = torch.max(corners1[:, :, 4,1][:, :, None], corners2[:, :, 4,1][:, None, :])
    height = (ymax - ymin).clamp(min=0)
    EPS = 1e-8
    
    idx = torch.arange(start=3, end=-1, step=-1, device=corners1.device)
    idx2 = torch.tensor([0,2], dtype=torch.int64, device=corners1.device)
    rect1 = corners1[:, :, idx, :]
    rect2 = corners2[:, :, idx, :]
    rect1 = rect1[:, :, :, idx2]
    rect2 = rect2[:, :, :, idx2]

    lt = torch.max(rect1[:, :, 1][:, :, None, :], rect2[:, :, 1][:, None, : ,:])
    rb = torch.min(rect1[:, :, 3][:, :, None, :], rect2[:, :, 3][:, None, : ,:])
    wh = (rb - lt).clamp(min=0)
    non_rot_inter_areas = wh[:, :, :, 0] * wh[:, :, :, 1]
    non_rot_inter_areas = non_rot_inter_areas.view(B, K1, K2)
    if nums_k2 is not None:
        for b in range(B):
            non_rot_inter_areas[b, :, nums_k2[b]:] = 0

    enclosing_vols = enclosing_box3d_vol(corners1, corners2)
    
    # vols of boxes
    vols1 = box3d_vol_tensor(corners1).clamp(min=EPS)
    vols2 = box3d_vol_tensor(corners2).clamp(min=EPS)
    
    sum_vols = vols1[:, :, None] + vols2[:, None, :]

    # filter malformed boxes
    good_boxes = (enclosing_vols > 2*EPS) * (sum_vols > 4*EPS)

    if rotated_boxes:
        inter_areas = torch.zeros((B, K1, K2), dtype=torch.float32)
        rect1 = rect1.cpu()
        rect2 = rect2.cpu()
        nums_k2_np = to_list_1d(nums_k2)
        non_rot_inter_areas_np = to_list_3d(non_rot_inter_areas)
        for b in range(B):
            for k1 in range(K1):
                for k2 in range(K2):
                    if nums_k2 is not None and k2 >= nums_k2_np[b]:
                        break
                    if non_rot_inter_areas_np[b][k1][k2] == 0:
                        continue
                    ##### compute volume of intersection
                    # inter = polygon_clip(rect1[b, k1], rect2[b, k2])
                    inter = polygon_clip_unnest(rect1[b, k1], rect2[b, k2])
                    # if inter is None:
                    # if len(inter) == 0:
                    #     # area = torch.zeros(1, dtype=torch.float32, device=inter_areas.device).squeeze()
                    #     # area = 0
                    #     continue
                    # else:
                    
                    if len(inter) > 0:
                        # inter = torch.stack(inter)
                        # xs = inter[:, 0]
                        # ys = inter[:, 1]
                        xs = torch.stack([x[0] for x in inter])
                        ys = torch.stack([x[1] for x in inter])
                        # area = poly_area_tensor(xs, ys)
                        inter_areas[b,k1,k2] = torch.abs(torch.dot(xs,torch.roll(ys,1))-torch.dot(ys,torch.roll(xs,1)))
        inter_areas.mul_(0.5)
    else:
        inter_areas = non_rot_inter_areas
  
    inter_areas = inter_areas.to(corners1.device)
    ### gIOU = iou - (1 - sum_vols/enclose_vol)
    inter_vols = inter_areas * height
    if return_inter_vols_only:
        return inter_vols

    union_vols = (sum_vols - inter_vols).clamp(min=EPS)
    ious = inter_vols / union_vols
    giou_second_term = - (1 - union_vols / enclosing_vols)
    gious = ious + giou_second_term
    gious *= good_boxes
    if nums_k2 is not None:
        mask = torch.zeros((B, K1, K2), device=height.device, dtype=torch.float32)
        for b in range(B):
            mask[b,:,:nums_k2[b]] = 1
        gious *= mask
    return gious

generalized_box3d_iou_tensor_jit = torch.jit.script(generalized_box3d_iou_tensor)

def enclosing_box3d_convex_hull(corners1, corners2, nums_k2, mask, enclosing_vols=None):
    B, K1 = corners1.shape[0], corners1.shape[1]
    _, K2 = corners2.shape[0], corners2.shape[1]
    if enclosing_vols is None:
        enclosing_vols = np.zeros((B, K1, K2)).astype(np.float32)
    for b in range(B):
        for k1 in range(K1):
            for k2 in range(K2):
                if nums_k2 is not None and k2 >= nums_k2[b]:
                    break
                if mask is not None and mask[b,k1,k2] <= 0:
                    continue

                hull = ConvexHull(np.vstack([corners1[b, k1], corners2[b, k2]]))
                enclosing_vols[b, k1, k2] = hull.volume
    return enclosing_vols

enclosing_box3d_convex_hull_numba = autojit(enclosing_box3d_convex_hull)
# enclosing_box3d_convex_hull_numba = enclosing_box3d_convex_hull

def generalized_box3d_iou_convex_hull_nondiff_tensor(corners1: torch.Tensor, corners2: torch.Tensor, nums_k2: torch.Tensor, rotated_boxes: bool = True):
    """
    Input:
        corners1: torch Tensor (B, K1, 8, 3), assume up direction is negative Y
        corners2: torch Tensor (B, K2, 8, 3), assume up direction is negative Y
        Assumes that the box is only rotated along Z direction
    Returns:
        B x K1 x K2 matrix of generalized IOU by approximating the boxes to be axis aligned
        The return IOU is differentiable
    """
    assert len(corners1.shape) == 4
    assert len(corners2.shape)== 4
    assert corners1.shape[2] == 8
    assert corners1.shape[3] == 3
    assert corners1.shape[0] == corners2.shape[0]
    assert corners1.shape[2] == corners2.shape[2]
    assert corners1.shape[3] == corners2.shape[3]

    B, K1 = corners1.shape[0], corners1.shape[1]
    _, K2 = corners2.shape[0], corners2.shape[1]
    EPS = 1e-8
    
    # vols of boxes
    vols1 = box3d_vol_tensor(corners1).clamp(min=EPS)
    vols2 = box3d_vol_tensor(corners2).clamp(min=EPS)
    
    sum_vols = vols1[:, :, None] + vols2[:, None, :]

    inter_vols = generalized_box3d_iou_tensor_jit(corners1, corners2, nums_k2, rotated_boxes, return_inter_vols_only=True)

    enclosing_vols = enclosing_box3d_vol(corners1, corners2)

    if rotated_boxes:
        corners1_np = corners1.detach().cpu().numpy()
        corners2_np = corners2.detach().cpu().numpy()
        mask = inter_vols.detach().cpu().numpy()
        nums_k2 = nums_k2.cpu().numpy()
        enclosing_vols_np = enclosing_vols.detach().cpu().numpy()
        enclosing_vols = enclosing_box3d_convex_hull_numba(corners1_np, corners2_np, nums_k2, mask, enclosing_vols_np)
        enclosing_vols = torch.from_numpy(enclosing_vols).to(corners1.device)
    
    union_vols = (sum_vols - inter_vols).clamp(min=EPS)
    ious = inter_vols / union_vols
    giou_second_term = - (1 - union_vols / enclosing_vols)
    gious = ious + giou_second_term
    good_boxes = (enclosing_vols > 2*EPS) * (sum_vols > 4*EPS)
    gious *= good_boxes
    if nums_k2 is not None:
        mask = torch.zeros((B, K1, K2), device=corners1.device, dtype=torch.float32)
        for b in range(B):
            mask[b,:,:nums_k2[b]] = 1
        gious *= mask
    return gious


def generalized_box3d_iou(corners1, corners2, nums_k2=None):
    """
    Input:
        corners1: torch Tensor (B, K1, 8, 3), assume up direction is negative Y
        corners2: torch Tensor (B, K2, 8, 3), assume up direction is negative Y
        mask: 
    Returns:
        B x K1 x K2 matrix of generalized IOU
    """
    # GenIOU = IOU - (C - sum_of_vols)/ C
    # where C = vol of convex_hull containing all points

    # degenerate boxes gives inf / nan results
    # so do an early check
    #TODO:
    assert corners1.ndim == 4
    assert corners2.ndim == 4
    assert corners1.shape[0] == corners2.shape[0]
    B, K1, _ , _ = corners1.shape
    _, K2, _, _ = corners2.shape

    gious = torch.zeros((B, K1, K2), dtype=torch.float32)

    corners1_np = corners1.detach().cpu().numpy()
    corners2_np = corners2.detach().cpu().numpy()

    for b in range(B):
        for i in range(K1):
            for j in range(K2):
                if nums_k2 is not None and j >= nums_k2[b]:
                    break
                iou, sum_of_vols = box3d_iou(corners1_np[b, i], corners2_np[b, j])

                hull = ConvexHull(np.vstack([corners1_np[b, i], corners2_np[b, j]]))
                C = hull.volume

                giou = iou - (C - sum_of_vols) / C
                gious[b, i, j] = giou
    return gious


# -----------------------------------------------------------
# Convert from box parameters to 
# -----------------------------------------------------------
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape)+[3,3]))
    c = np.cos(t)
    s = np.sin(t)
    output[...,0,0] = c
    output[...,0,2] = s
    output[...,1,1] = 1
    output[...,2,0] = -s
    output[...,2,2] = c
    return output


def get_3d_box(box_size, heading_angle, center):
    ''' box_size is array(l,w,h), heading_angle is radius clockwise from pos x axis, center is xyz of box center
        output (8,3) array for 3D box cornders
        Similar to utils/compute_orientation_3d
    '''
    R = roty(heading_angle)
    l,w,h = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)
    return corners_3d

def get_3d_box_batch(box_size, heading_angle, center):
    ''' box_size: [x1,x2,...,xn,3] -- box dimensions without flipping [X, Y, Z] -- l, w, h
        heading_angle: [x1,x2,...,xn] -- theta in radians
        center: [x1,x2,...,xn,3] -- center point has been flipped to camera axis [X, -Z, Y]
    Return:
        [x1,x3,...,xn,8,3]
    '''
    input_shape = heading_angle.shape
    R = roty_batch(heading_angle)
    l = np.expand_dims(box_size[...,0], -1) # [x1,...,xn,1]
    w = np.expand_dims(box_size[...,1], -1)
    h = np.expand_dims(box_size[...,2], -1)
    corners_3d = np.zeros(tuple(list(input_shape)+[8,3]))
    corners_3d[...,:,0] = np.concatenate((l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2), -1)
    corners_3d[...,:,1] = np.concatenate((h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2), -1)
    corners_3d[...,:,2] = np.concatenate((w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape)+1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d

def roty_batch_tensor(t):
    input_shape = t.shape
    output = torch.zeros(tuple(list(input_shape)+[3,3]), dtype=torch.float32, device=t.device)
    c = torch.cos(t)
    s = torch.sin(t)
    output[...,0,0] = c
    output[...,0,2] = s
    output[...,1,1] = 1
    output[...,2,0] = -s
    output[...,2,2] = c
    return output

def flip_axis_to_camera_tensor(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = torch.clone(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2


def get_3d_box_batch_tensor(box_size, heading_angle, center):
    assert isinstance(box_size, torch.Tensor)
    assert isinstance(heading_angle, torch.Tensor)
    assert isinstance(center, torch.Tensor)

    reshape_final = False
    if heading_angle.ndim == 2:
        assert box_size.ndim == 3
        assert center.ndim == 3
        bsize = box_size.shape[0]
        nprop = box_size.shape[1]
        box_size = box_size.view(-1, box_size.shape[-1])
        heading_angle = heading_angle.view(-1)
        center = center.reshape(-1, 3)
        reshape_final = True
    
    input_shape = heading_angle.shape
    R = roty_batch_tensor(heading_angle)
    l = torch.unsqueeze(box_size[...,0], -1) # [x1,...,xn,1]
    w = torch.unsqueeze(box_size[...,1], -1)
    h = torch.unsqueeze(box_size[...,2], -1)
    corners_3d = torch.zeros(tuple(list(input_shape)+[8,3]), device=box_size.device, dtype=torch.float32)
    corners_3d[...,:,0] = torch.cat((l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2), -1)
    corners_3d[...,:,1] = torch.cat((h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2), -1)
    corners_3d[...,:,2] = torch.cat((w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape)+1, len(input_shape)]
    corners_3d = torch.matmul(corners_3d, R.permute(tlist))
    corners_3d += torch.unsqueeze(center, -2)
    if reshape_final:
        corners_3d = corners_3d.reshape(bsize, nprop, 8, 3)
    return corners_3d


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


if __name__=='__main__':
    
    # Function for polygon ploting
    import matplotlib
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.pyplot as plt
    def plot_polys(plist,scale=500.0):
        fig, ax = plt.subplots()
        patches = []
        for p in plist:
            poly = Polygon(np.array(p)/scale, True)
            patches.append(poly)

    pc = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.5)
    colors = 100*np.random.rand(len(patches))
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    plt.show()
 
    # Demo on ConvexHull
    points = np.random.rand(30, 2)   # 30 random points in 2-D
    hull = ConvexHull(points)
    # **In 2D "volume" is is area, "area" is perimeter
    print(('Hull area: ', hull.volume))
    for simplex in hull.simplices:
        print(simplex)

    # Demo on convex hull overlaps
    sub_poly = [(0,0),(300,0),(300,300),(0,300)]
    clip_poly = [(150,150),(300,300),(150,450),(0,300)] 
    inter_poly = polygon_clip(sub_poly, clip_poly)
    print(poly_area(np.array(inter_poly)[:,0], np.array(inter_poly)[:,1]))
    
    # Test convex hull interaction function
    rect1 = [(50,0),(50,300),(300,300),(300,0)]
    rect2 = [(150,150),(300,300),(150,450),(0,300)] 
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    print((inter, area))
    if inter is not None:
        print(poly_area(np.array(inter)[:,0], np.array(inter)[:,1]))
    
    print('------------------')
    rect1 = [(0.30026005199835404, 8.9408694211408424), \
             (-1.1571105364358421, 9.4686676477075533), \
             (0.1777082043006144, 13.154404877812102), \
             (1.6350787927348105, 12.626606651245391)]
    rect1 = [rect1[0], rect1[3], rect1[2], rect1[1]]
    rect2 = [(0.23908745901608636, 8.8551095691132886), \
             (-1.2771419487733995, 9.4269062966181956), \
             (0.13138836963152717, 13.161896351296868), \
             (1.647617777421013, 12.590099623791961)]
    rect2 = [rect2[0], rect2[3], rect2[2], rect2[1]]
    plot_polys([rect1, rect2])
    inter, area = convex_hull_intersection(rect1, rect2)
    print((inter, area))
