import cv2
import numpy as np
import trimesh
from matplotlib.tri import Triangulation
import yaml
import argparse
import os
import sys

class MapConverter():
    def __init__(self, map_dir, export_dir, threshold=105, height=0.6):
        
        self.threshold = threshold
        self.height = height
        self.export_dir = export_dir
        self.map_dir = map_dir

    def map_callback(self):
        
        map_array = cv2.imread(self.map_dir)
        map_array = cv2.flip(map_array, 0)
        print(f'loading map file: {self.map_dir}')
        try:
            map_array = cv2.cvtColor(map_array, cv2.COLOR_BGR2GRAY)
        except cv2.error as err:
            print(err, "Conversion failed: Invalid image input, please check your file path")    
            sys.exit()
        info_dir = self.map_dir.replace('pgm','yaml')

        with open(info_dir, 'r') as stream:
            map_info = yaml.load(stream, Loader=yaml.FullLoader) 
        
        # set all -1 (unknown) values to 0 (unoccupied)
        map_array[map_array < 0] = 0
        tresh_map = self.get_occupied_regions(map_array)
        print('Processing...')
        mesh = self.cells_to_mesh(tresh_map, map_info)

        if not self.export_dir.endswith('/'):
            self.export_dir = self.export_dir + '/'
        file_dir = self.export_dir + map_info['image'].replace('pgm','stl')
        print(f'export file: {file_dir}')
        
        with open(file_dir, 'wb') as f:
            mesh.export(f, "stl")
    
    def get_occupied_regions(self, map_array):
        map_array = map_array.astype(np.uint8)
        _, thresh_map = cv2.threshold(map_array, self.threshold, 100, cv2.THRESH_BINARY)
        # Now get the indices of occupied cells directly from the thresholded map
        occupied_cells = np.column_stack(np.where(thresh_map != 100))  # Get coordinates of non-zero cells
        
        return occupied_cells
    
    def cells_to_mesh(self, occupied_cells, metadata):
        height = np.array([0, 0, self.height])  # The height of the prisms
        meshes = []
        
        for cell in occupied_cells:
            y, x = cell  # Extract row and column for each occupied cell
            
            # Define the four corners of the occupied cell
            new_vertices = [
                self.coords_to_loc((x - 0.5, y - 0.5), metadata),
                self.coords_to_loc((x - 0.5, y + 0.5), metadata),
                self.coords_to_loc((x + 0.5, y - 0.5), metadata),
                self.coords_to_loc((x + 0.5, y + 0.5), metadata)
            ]
            
            # Add height to create the 3D vertices (top and bottom faces of the prism)
            vertices = []
            vertices.extend(new_vertices)
            vertices.extend([v + height for v in new_vertices])
            
            # Define the faces of the prism (6 faces for a rectangular prism)
            faces = [
                [0, 2, 4], [4, 2, 6],  # Front faces
                [1, 2, 0], [3, 2, 1],  # Back faces
                [5, 0, 4], [1, 0, 5],  # Bottom faces
                [3, 7, 2], [7, 6, 2],  # Top faces
                [7, 4, 6], [5, 4, 7],  # Left faces
                [1, 5, 3], [7, 3, 5]   # Right faces
            ]
            
            # Create the mesh for the prism
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if not mesh.is_volume:
                mesh.fix_normals()  # Ensure correct normals if the mesh is not solid
            
            meshes.append(mesh)  # Add the mesh for this occupied cell
        
        # Combine all meshes into a single mesh
        combined_mesh = trimesh.util.concatenate(meshes)
        combined_mesh.update_faces(combined_mesh.unique_faces())
        
        return combined_mesh

    def coords_to_loc(self,coords, metadata):
        x, y = coords
        loc_x = x * metadata['resolution'] + metadata['origin'][0]
        loc_y = y * metadata['resolution'] + metadata['origin'][1]
        # TODO: transform (x*res, y*res, 0.0) by Pose map_metadata.origin
        # instead of assuming origin is at z=0 with no rotation wrt map frame
        return np.array([loc_x, loc_y, 0.0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--map_dir', type=str, required=True,
        help='File name of the map to convert'
    )

    parser.add_argument(
        '--export_dir', type=str, default=os.path.abspath('.'),
        help='Mesh output directory'
    )

    option = parser.parse_args()

    Converter = MapConverter(option.map_dir, option.export_dir)
    Converter.map_callback()
    print('Conversion Done')
