import open3d as o3d
import numpy as np
import trimesh

def clear_bottom_points(input_path, output_path, height_threshold=0.05):
    # Read the mesh file
    mesh = o3d.io.read_triangle_mesh(input_path)
    
    # Get vertices as numpy array
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Create mask for vertices above threshold
    mask = vertices[:, 2] > height_threshold
    
    # Get indices mapping
    new_indices = np.cumsum(mask) - 1
    
    # Filter vertices
    new_vertices = vertices[mask]
    
    # Update triangles to use new vertex indices
    valid_triangles = []
    for triangle in triangles:
        if mask[triangle[0]] and mask[triangle[1]] and mask[triangle[2]]:
            new_triangle = [new_indices[i] for i in triangle]
            valid_triangles.append(new_triangle)
    
    # Create new mesh
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(valid_triangles)
    
    # Save the modified mesh
    o3d.io.write_triangle_mesh(output_path, new_mesh)

def clear_mesh_cache():
    """Clear trimesh's cache to ensure fresh loading of meshes"""
    trimesh.util.reset_log()
    trimesh.cache.clear()

if __name__ == "__main__":
    # Example usage
    input_file = "meshes/5a_simplified.ply"  # Replace with your input file path
    output_file = "meshes/5a_clear_simplified.ply"  # Replace with desired output file path
    clear_bottom_points(input_file, output_file)