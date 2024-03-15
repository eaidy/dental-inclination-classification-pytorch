# 3D File utility services
# The architecture intended to process three different snapshots from arbitrary different angles,
# First consideration is to employ the same number of different neural network archs as much as the number of snapshots taken,
# After evaluation for each network, the output values will gathered from each network and avarage value of them will decide the classification.
# This factory class is going to produce snapshots from given point of views and saves them into the specified dataset directory.

import glob
import os
import pyvista as pv

class SnapshotFactory:
    def __init__(self, 
                stl_dataset_dir,
                output_dataset_dir="/Users/eaidy/Repos/ML/inclination-classification-pytorch/dataset_2d_stl", 
                view_points=[(-150, -100, 0), (50, 50, -150), (-50, -150, 20)], 
                focal_point=(0, 0, 0), 
                view_up=(0, 0, 1)):
        
        self.stl_dataset_dir = stl_dataset_dir
        self.view_points = view_points
        self.focal_point = focal_point
        self.view_up = view_up
        self.output_dataset_dir = output_dataset_dir
    
    # Generates all snapshots for all labels
    def generate_snapshots_all(self):         
        stl_files_dict = self._get_stl_files_dict()

        # STL file directories not actual files
        for view_point in self.view_points:
            vp_index = self.view_points.index(view_point)
            self._generate_snapshots_viewpoint(stl_files_dict, view_point, vp_index)
            
    # generates all snapshots for all labels for a specific viewpoint
    def _generate_snapshots_viewpoint(self, stl_files_dict, view_point, vp_index):
        for stl_label, stl_files in stl_files_dict.items():
            self._generate_snapshots_label(stl_label, stl_files, view_point, vp_index)

    # Generates all snapshots for spesific label
    def _generate_snapshots_label(self, stl_label, stl_files, view_point, vp_index):
        for stl_file in stl_files:
            stl_index = stl_files.index(stl_file)
            self._generate_snapshot_single(
                                    stl_dir=stl_file, 
                                    view_point=view_point, 
                                    snapshot_index=stl_index, 
                                    view_point_index=vp_index, 
                                    label=stl_label,
                                    zoom=1.0)
            
    # Generates a single snapshot for a given viewpoint      
    def _generate_snapshot_single(self, stl_dir, view_point, snapshot_index, view_point_index, label, zoom=1.0):
        mesh = pv.read(stl_dir)

        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh, color='white')

        camera_position = view_point  # x, y, z position of the camera
        focal_point = self.focal_point  # x, y, z
        view_up = self.view_up  # Typically (0, 1, 0) or (0, 0, 1) depending on the coordinate system

        plotter.camera.position = camera_position
        plotter.camera.focal_point = focal_point
        plotter.camera.view_up = view_up

        plotter.camera.zoom(zoom)

        output_dir = os.path.join(self.output_dataset_dir, f'viewpoint_{view_point_index}', label)
        os.makedirs(output_dir, exist_ok=True)  # This creates the directory if it doesn't exist

        output_filename = os.path.join(output_dir, f'{view_point_index}_{snapshot_index}.png')
        #output_filename = self.output_dataset_dir + f'/viewpoint_{view_point_index}/{label}/{view_point_index}_{snapshot_index}.png'
       
        plotter.screenshot(output_filename)

    def _get_stl_files_dict(self): 
        dataset_path = self.stl_dataset_dir

        stl_dict = {}
        for label_path in glob.glob(os.path.join(dataset_path, '*', '')):
            label = os.path.basename(os.path.dirname(label_path)) 
           
            stl_files = glob.glob(os.path.join(label_path, '*.stl'))
            if stl_files:
                stl_dict[label] = stl_files
        return stl_dict