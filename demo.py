"""Fuse 1000 RGB-D images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""

import time

import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
import fusion
from numba import jit
#import icp.basicICP as icp
import Generalized_ICP.Generalized_ICP

def configureCamera():
  k4a = PyK4A(
    Config(
      color_resolution=pyk4a.ColorResolution.RES_720P,
      depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
      synchronized_images_only=True
    )
  )
  return k4a


def depth_image_to_point_cloud(map, h, w, K):
  points = np.ndarray((h * w, 3))
  return depth_image_to_point_cloud_numba(map, h, w, K, points)


@jit(nopython=True)
def depth_image_to_point_cloud_numba(map, h, w, K, points):
  for u, row in enumerate(map):
    for v, depth in enumerate(row):

      u_ndc = u / w
      v_ndc = v / h

      u_screen = u_ndc * w - w / 2
      v_screen = v_ndc * h - h / 2

      fx = K[0][0]
      fy = K[1][1]
      fxinv = 1.0 / fx
      fyinv = 1.0 / fy

      cx = K[0][2]
      cy = K[1][2]

      points[u * w + v] = [(u_screen - cx) * fxinv * depth, (v_screen - cy) * fyinv * depth, -depth]
  return points


def icp(curr, prev, method):
  R, T, rms = Generalized_ICP.Generalized_ICP.ICP(curr, prev, method)
  print(rms)
  pose = np.column_stack((R, T))
  np.vstack((pose, np.array([0, 0, 0, 1])))
  return pose


if __name__ == "__main__":
  # ======================================================================================================== #
  # (Optional) This is an example of how to compute the 3D bounds
  # in world coordinates of the convex hull of all camera view
  # frustums in the dataset
  # ======================================================================================================== #
  print("Estimating voxel volume bounds...")
  n_imgs = 1000
  cam_intr = np.loadtxt("data_original/camera-intrinsics.txt", delimiter=' ')
  vol_bnds = np.zeros((3,2))
  for i in range(n_imgs):
    # Read depth image and camera pose
    depth_im = cv2.imread("data_original/frame-%06d.depth.png"%(i),-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    cam_pose = np.loadtxt("data_original/frame-%06d.pose.txt"%(i))  # 4x4 rigid transformation matrix

    # Compute camera view frustum and extend convex hull
    view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
    vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
    vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
  # ======================================================================================================== #
  print(vol_bnds)
  # ======================================================================================================== #
  # Integrate
  # ======================================================================================================== #
  # Initialize voxel volume
  print("Initializing voxel volume...")
  tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

  # Loop through RGB-D images and fuse them together
  t0_elapse = time.time()

  # i == 0:
  color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg"%(0)), cv2.COLOR_BGR2RGB)
  depth_im = cv2.imread("data/frame-%06d.depth.png"%(0),-1).astype(float)
  depth_im /= 1000.
  depth_im[depth_im == 65.535] = 0
  cam_pose = np.loadtxt("data_original/frame-%06d.pose.txt" % (0))

  height = depth_im.shape[0]
  width = depth_im.shape[1]
  camera_intrinsics = np.loadtxt("data/camera-intrinsics.txt")
  curr = Generalized_ICP.Generalized_ICP.Point_cloud()
  curr.init_from_points(depth_image_to_point_cloud(depth_im, height, width, camera_intrinsics))


  for i in range(n_imgs):
    print("Fusing frame %d/%d"%(i+1, n_imgs))

    # Read RGB-D image and camera pose
    color_image = cv2.cvtColor(cv2.imread("data/frame-%06d.color.jpg"%(i)), cv2.COLOR_BGR2RGB)
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(i),-1).astype(float)
    depth_im /= 1000.
    depth_im[depth_im == 65.535] = 0

    prev = curr
    curr.init_from_points(depth_image_to_point_cloud(depth_im, height, width, camera_intrinsics))
    cam_transformation = icp(curr, prev, "point2plane")
    cam_pose = cam_transformation @ cam_pose

    with open("data/frame-%06d.pose.txt" % (i), 'w') as pose_file:
      for row in cam_pose:
        pose_file.write(str(row[0]) + ' ' + str(row[1]) + ' ' + str(row[2]) + ' ' + str(row[3]) + '\n')

    # Integrate observation into voxel volume (assume color aligned with depth)
    tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

  fps = n_imgs / (time.time() - t0_elapse)
  print("Average FPS: {:.2f}".format(fps))

  # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving mesh to mesh.ply...")
  verts, faces, norms, colors = tsdf_vol.get_mesh()
  fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

  # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
  print("Saving point cloud to pc.ply...")
  point_cloud = tsdf_vol.get_point_cloud()
  fusion.pcwrite("pc.ply", point_cloud)