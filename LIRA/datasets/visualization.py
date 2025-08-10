import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from mpl_toolkits.mplot3d import Axes3D
import torch
from skimage import measure
import trimesh
from matplotlib import cm
from scipy.spatial import Delaunay


def plot_voxels(voxels, title="Voxel Grid"):
    """ Visualize 3D voxel data using pyvista """
    grid = pv.UniformGrid()

    grid.dimensions = np.array(voxels.shape) + 1

    grid.origin = (0, 0, 0)  # The bottom left corner of the data set

    grid.spacing = (1, 1, 1)  # These are the cell sizes along each axis

    grid.cell_data["values"] = voxels.flatten(order="F")  # Flatten the array!

    plotter = pv.Plotter()
    plotter.add_volume(grid, scalars="values", opacity='sigmoid', show_scalar_bar=False)
    plotter.set_background("white")
    plotter.camera_position = 'iso'
    plotter.add_title(title)
    plotter.show()


def visualize_mesh(input_A=None, input_B=None, type='xyz', use_path=False, out_file=None):

    # A = np.load('datasets/scannet/panoptic_info/scene0000_00_vert.npy')  xyz(N, 3) or xyzrgb(N, 6)
    # B = np.load('datasets/scannet/panoptic_info/scene0000_00_sem_label.npy')  class_id(N,)
    # C = np.load('datasets/scannet/panoptic_info/scene0000_00_ins_label.npy')  instance_id(N,)

    if use_path:
        A = np.load(input_A)
    else:
        A = input_A
    N = A.shape[0]
    A = A.reshape(N, -1)
    if type == 'xyz':
        points = pv.PolyData(A[:, :3])

        gray_color = np.array([0.5, 0.5, 0.5])
        points.point_data['gray_colors'] = np.tile(gray_color, (N, 1))

        plotter = pv.Plotter()
        plotter.add_points(points, scalars='gray_colors', rgb=True, point_size=5)

        plotter.add_axes(interactive=True)
        plotter.show_grid()

        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        # Set the position of the coordinate axes and fix the origin at the lower left corner
        plotter.view_vector([-1, -1, 0]) 
        plotter.reset_camera()  

        plotter.show(title='PointCloud Gray Visualization')

    elif type == 'rgb':  # 0-255
        rgb_values = A[:, 3:]  
        non_zero_indices = ~np.all(rgb_values == 0, axis=1) 
        A = A[non_zero_indices]  

        points_rgb = pv.PolyData(A[:, :3])
        points_rgb['rgb'] = A[:, 3:6] / 255 

        plotter = pv.Plotter()
        plotter.add_points(points_rgb, rgb=True, point_size=5)

        plotter.add_axes(interactive=True)
        plotter.show_grid()

        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        plotter.view_vector([-1, -1, 0])  
        plotter.reset_camera()  

        plotter.show(title='Point Cloud with RGB Colors')

    elif type == 'semantic':
        if use_path:
            B = np.load(input_B)
        else:
            B = input_B
        B = B.reshape(N, -1)

        A = A[B.reshape(-1) > 0]
        B = B[B.reshape(-1) > 0]

        # 
        # manual_colors = np.array([
        #     [0, 0, 1],  
        #     [1, 0, 0],  
        #     [0, 1, 0]  
        # ])
        #
        # 
        # num_classes = 51
        # colors = plt.get_cmap('Set3', num_classes)  
        # generated_colors = [colors(i / num_classes) for i in range(num_classes)]
        # generated_colors = np.array(generated_colors)[:, :3] 
        #
        # semantic_colors = np.vstack((manual_colors, generated_colors[3:num_classes]))
        #
        # semantic_class_colors = np.array([semantic_colors[i[0]] for i in B])

        semantic_class_colors = np.random.rand(51, 3)  
        semantic_class_colors = semantic_class_colors[B.flatten()]  

        points_semantic = pv.PolyData(A[:, :3])
        points_semantic['semantic_colors'] = semantic_class_colors

        plotter = pv.Plotter()
        plotter.add_points(points_semantic, rgb=True, point_size=5)

        plotter.add_axes(interactive=True)
        plotter.show_grid()

        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        plotter.view_vector([-1, -1, 0])
        plotter.reset_camera()  

        plotter.show(title='Point Cloud with Semantic Labels')

    elif type == 'instance':
        if use_path:
            B = np.load(input_B)
        else:
            B = input_B
        B = B.reshape(N, -1)

        A = A[B.reshape(-1) != 0]
        B = B[B.reshape(-1) != 0]

        points = pv.PolyData(A[:, :3])


        instance_colors = np.random.rand(np.max(B)+1, 3)  
        instance_colors = instance_colors[B.flatten()]  

        points.point_data['instance_colors'] = instance_colors


        plotter = pv.Plotter()
        plotter.add_points(points, scalars='instance_colors', rgb=True, point_size=5)

        plotter.add_axes(interactive=True)
        plotter.show_grid()

        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)

        plotter.view_vector([-1, -1, 0])
        plotter.reset_camera()  

        plotter.show(title='Instance Segmentation Result')

    elif type == 'tsdf':
        if use_path:
            B = np.load(input_B)
        else:
            B = input_B
        B = B.reshape(N, -1)

        A = A[B.reshape(-1) > 0]
        B = B[B.reshape(-1) > 0]

        # normalized_tsdf = (B - np.min(B)) / (np.max(B) - np.min(B)) * 2 - 1
        normalized_tsdf = B

        grid = pv.PolyData(A[:, :3]) 
   
        grid.point_data['tsdf'] = normalized_tsdf.flatten()


        plotter = pv.Plotter()
        plotter.add_mesh(grid, scalars='tsdf', cmap='plasma')

        plotter.add_axes(interactive=True)
        plotter.show_grid()

        origin = pv.Sphere(radius=1.0, center=(0, 0, 0), direction=(0, 0, 1))
        plotter.add_mesh(origin, color='red', show_edges=False)


        plotter.view_vector([-1, -1, 0])  
        plotter.reset_camera() 

        plotter.show(title='TSDF Visualization')

        # Save the grid to a PLY file
        if out_file is not None:
            grid.save(out_file)

    else:
        pass


def sparse_to_dense_torch(locs, values, dim, default_val, device, final_tsdf=False):

    if final_tsdf:  # The final tsdf needs to be filled with the nearest neighbors -1 and 1
        # default_val = 100
        # dense = torch.full([dim[0], dim[1], dim[2]], default_val, device=device).to(values.dtype)
        # if locs.shape[0] > 0:
        #     dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values
        #
        # mask = dense == default_val  # The value that needs to be replaced
        #
        # dense, mask = dense.cpu().numpy(), mask.cpu().numpy()
        # distance, indices = ndimage.distance_transform_edt(mask, return_indices=True)  # Calculate the distance transformation to obtain the distance from each point to the nearest point that does not need to be replaced
        # nearest_values = dense[tuple(indices)]  # Use the indices mapping to find the value of the nearest point to each point that does not need to be replaced
        # dense[mask] = np.where(nearest_values[mask] >= 0, 1, -1)  # The replacement value is determined based on the value of the nearest point. Positive numbers are replaced with 1, and negative numbers with -1
        # dense = torch.from_numpy(dense).to(device)

        dense = torch.full([dim[0], dim[1], dim[2]], default_val, device=device).to(values.dtype)
        if locs.shape[0] > 0:
            dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values

    else:
        dense = torch.full([dim[0], dim[1], dim[2]], float(default_val), device=device).to(values.dtype)
        if locs.shape[0] > 0:
            dense[locs[:, 0], locs[:, 1], locs[:, 2]] = values

    return dense


def label2mesh(coords, tsdf, dim_list, label, save_path):
    tsdf = tsdf.view(-1).detach()
    tsdf_volume = sparse_to_dense_torch(coords.long(), tsdf, dim_list, 1, tsdf.device, final_tsdf=True)  # The final tsdf needs to be filled with -1 and 1
    tsdf_volume = torch.clamp(tsdf_volume, -1, 1)
    tsdf_volume = tsdf_volume.cpu().numpy()

    label = label.view(-1).detach()
    label_volume = sparse_to_dense_torch(coords.long(), label, dim_list, 0, label.device, final_tsdf=False)  # 0 pad
    label_volume = label_volume.cpu().numpy()

    # Step 1: extract mesh
    verts, faces, norms, vals = measure.marching_cubes(tsdf_volume, level=0)  
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=norms)

    # Step 2: Map the label to the vertices of the mesh
    # Use nearest neighbor interpolation to find the corresponding coordinates of each vertex in label_volume and obtain the label
    verts_scaled = (verts / np.array(tsdf_volume.shape)) * np.array(label_volume.shape)  # Scale the vertex to the range of label_volume
    verts_indices = np.round(verts_scaled).astype(int)  # Take the nearest integer index
    verts_indices = np.clip(verts_indices, 0, np.array(label_volume.shape) - 1)  # Prevent crossing boundaries
    vertex_labels = label_volume[verts_indices[:, 0], verts_indices[:, 1], verts_indices[:, 2]]

    # Step 3: Assign a color to each label
    unique_labels = np.unique(vertex_labels)

    colormap = cm.get_cmap('tab20', len(unique_labels)) 
    
    label_to_color = {}
    for i, label in enumerate(unique_labels):
        if label == 0:
            label_to_color[label] = np.array([192, 192, 192], dtype=np.uint8)  
        elif label == 1:
            label_to_color[label] = np.array([173, 216, 230], dtype=np.uint8)  
        elif label == 2:
            label_to_color[label] = np.array([216, 191, 216], dtype=np.uint8) 
        elif label == 3:
            label_to_color[label] = np.array([255, 255, 0], dtype=np.uint8)   
        elif label == 4:
            label_to_color[label] = np.array([0, 255, 0], dtype=np.uint8)  
        else:
            label_to_color[label] = (np.array(colormap(i)[:3]) * 255).astype(np.uint8)
    
    
    vertex_colors = np.array([label_to_color[label] for label in vertex_labels])

    mesh.visual.vertex_colors = vertex_colors
    mesh.export(save_path)

    print("mesh save to ", save_path)

colors_list = [
    [230, 230, 230], 
    [255, 0, 0],      
    [0, 255, 0],     
    [0, 0, 255],    
    [255, 255, 0],    
    [0, 255, 255],   
    [255, 0, 255],   
    [255, 165, 0],   
    [128, 0, 128],    
    [0, 128, 128],   
    [128, 128, 0],   
    [0, 0, 128],     
    [255, 192, 203],  
    [255, 69, 0],    
    [75, 0, 130],    
    [255, 99, 71],  
    [34, 139, 34],   
    [0, 0, 139],      
    [184, 134, 11],   
    [255, 215, 0],   
]

def save_coords_with_labels_to_ply(coords, labels, filename):
    assert coords.ndim == 2 and coords.shape[1] == 3
    assert labels.ndim == 1 and labels.shape[0] == coords.shape[0]
    
    num_labels = np.max(labels) + 1  
    colors = np.zeros((labels.shape[0], 3), dtype=int)  

    for i in range(labels.shape[0]):
        label = labels[i]
        if label == 0:
            colors[i] = colors_list[0] 
        else:
            colors[i] = colors_list[(label - 1) % 19 + 1] 
    
    num_points = coords.shape[0]
    
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with open(filename, 'w') as ply_file:
        ply_file.write(header)

        for i in range(num_points):
            ply_file.write(f"{coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f} "
                           f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")


if __name__ == '__main__':
    path_A = 'datasets/scannet/panoptic_info/scene0000_00_vert.npy'
    path_B = 'datasets/scannet/panoptic_info/scene0000_00_sem_label.npy'
    path_C = 'datasets/scannet/panoptic_info/scene0000_00_ins_label.npy'

    # visualize_mesh(path_A, type='xyz', use_path=True)
    # visualize_mesh(path_A, type='rgb', use_path=True)
    # visualize_mesh(path_A, path_B, type='semantic', use_path=True)
    # visualize_mesh(path_A, path_C, type='instance', use_path=True)
    visualize_mesh(path_A, path_B, type='tsdf', use_path=True)


