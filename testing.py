import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Sample data (replace with your actual `nodes`, `tetrahedrons`, and `tets_to_faces`)
nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
tetrahedrons = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
tets_to_faces = {
    0: np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
    1: np.array([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]])
}

def plot_tetrahedral_mesh(nodes, tetrahedrons, tets_to_faces):
    # Set up the figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each tetrahedron
    for tet_index, faces in tets_to_faces.items():
        face_coords = [nodes[face] for face in faces]
        ax.add_collection3d(Poly3DCollection(face_coords, color="cyan", alpha=0.5, edgecolor="k"))

    # Set the plot limits and aspect ratio
    ax.set_aspect('auto')
    ax.set_xlim(np.min(nodes[:, 0]), np.max(nodes[:, 0]))
    ax.set_ylim(np.min(nodes[:, 1]), np.max(nodes[:, 1]))
    ax.set_zlim(np.min(nodes[:, 2]), np.max(nodes[:, 2]))

    # Display the interactive plot
    plt.ion()  # Enable interactive mode
    plt.show(block=True)

# Call the function with your actual data
plot_tetrahedral_mesh(nodes, tetrahedrons, tets_to_faces)
