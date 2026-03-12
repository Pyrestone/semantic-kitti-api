#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from auxiliary.vispy_manager import VispyManager
from datetime import datetime
import json
import numpy as np

class LaserScanComp(VispyManager):
  """Class that creates and handles a side-by-side pointcloud comparison"""

  def __init__(self, scans, scan_names, label_names, offset=0, images=True, instances=False, link=False, camera_state:str=None):
    super().__init__(offset, len(scan_names), images, instances)
    self.scan_a_view = None
    self.scan_a_vis = None
    self.scan_b_view = None
    self.scan_b_vis = None
    self.inst_a_view = None
    self.inst_a_vis = None
    self.inst_b_view = None
    self.inst_b_vis = None
    self.img_a_view = None
    self.img_a_vis = None
    self.img_b_view = None
    self.img_b_vis = None
    self.img_inst_a_view = None
    self.img_inst_a_vis = None
    self.img_inst_b_view = None
    self.img_inst_b_vis = None
    self.scan_a, self.scan_b, self.scan_c = scans
    self.scan_names = scan_names
    self.label_a_names, self.label_b_names = label_names
    self.link = link
    self.reset()
    self.update_scan()
    if camera_state is not None:
      self.load_camera_state(camera_state)

  def reset(self):
    """prepares the canvas(es) for the visualizer"""
    self.scan_a_view, self.scan_a_vis = super().add_viewbox(0, 0)
    self.scan_b_view, self.scan_b_vis = super().add_viewbox(0, 1)
    self.scan_c_view, self.scan_c_vis = super().add_viewbox(0, 2)

    if self.link:
      self.scan_a_view.camera.link(self.scan_b_view.camera)
      self.scan_c_view.camera.link(self.scan_a_view.camera)

    if self.images:
      self.img_a_view, self.img_a_vis = super().add_image_viewbox(0, 0)
      self.img_b_view, self.img_b_vis = super().add_image_viewbox(1, 0)
      self.img_c_view, self.img_c_vis = super().add_image_viewbox(2, 0)

  def update_scan(self):
    """updates the scans, images and instances"""
    self.scan_a.open_scan(self.scan_names[self.offset])
    self.scan_a.open_label(self.label_a_names[self.offset])
    self.scan_a.colorize()
    self.scan_a_vis.set_data(self.scan_a.points,
                          face_color=self.scan_a.sem_label_color[..., ::],
                          edge_color=self.scan_a.sem_label_color[..., ::],
                          size=3.5,edge_width=0.0)

    self.scan_b.open_scan(self.scan_names[self.offset])
    self.scan_b.open_label(self.label_b_names[self.offset])
    self.scan_b.colorize()
    self.scan_b_vis.set_data(self.scan_b.points,
                          face_color=self.scan_b.sem_label_color[..., ::],
                          edge_color=self.scan_b.sem_label_color[..., ::],
                          size=3.5,edge_width=0.0)


    diff_labels = np.where(self.scan_a.sem_label==self.scan_b.sem_label, 1, 2)
    diff_labels = np.where(self.scan_a.sem_label==0, 0, diff_labels)
    self.scan_c.open_scan(self.scan_names[self.offset])
    
    self.scan_c.sem_label = diff_labels


    self.scan_c.colorize()
    self.scan_c_vis.set_data(self.scan_c.points,
                          face_color=self.scan_c.sem_label_color[..., ::-1],
                          edge_color=self.scan_c.sem_label_color[..., ::-1],
                          size=3.5,edge_width=0.0)

    if self.images:
      self.img_a_vis.set_data(self.scan_a.proj_sem_color[..., ::])
      self.img_a_vis.update()
      self.img_b_vis.set_data(self.scan_b.proj_sem_color[..., ::])
      self.img_b_vis.update()
      self.img_c_vis.set_data(self.scan_c.proj_sem_color[..., ::-1])
      self.img_c_vis.update()


  def save_camera(self):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"camera_state_{timestamp}.json"
    camera_state = self.scan_a_view.camera.get_state()
    state = {"camera_state": camera_state}
    

    with open(filename, "w") as f:
      json.dump(state, f, indent=4)

    print(f"Saved camera state as: {filename}")

  def load_camera_state(self, camera_state):
    with open(camera_state, "r") as file:
      state = json.load(file)
    
    self.scan_a_view.camera.set_state(state["camera_state"])

