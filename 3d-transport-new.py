# import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import animation, cm
import gmsh
import meshio
# import sys
from collections import defaultdict, deque
import networkx as nx
# from IPython.display import HTML
import pyvista as pv

# Function to create a 3D tetrahedral cubic mesh in gmsh
def create_cubic_mesh(l, ms, mesh_fname, visualize_mesh=False):
    # Initialize the gmsh API
    gmsh.initialize()

    # Create a new model
    gmsh.model.add("cube")

    # Define the length of the cube sides
    # L = l # Default to 1.0

    # Define characteristic lengths
    # mesh_size = ms  # You can adjust this to control the element size (default to 0.9)

    # Define points of the cube with the specified mesh size
    p1 = gmsh.model.geo.addPoint(0, 0, 0, ms)
    p2 = gmsh.model.geo.addPoint(l, 0, 0, ms)
    p3 = gmsh.model.geo.addPoint(l, l, 0, ms)
    p4 = gmsh.model.geo.addPoint(0, l, 0, ms)
    p5 = gmsh.model.geo.addPoint(0, 0, l, ms)
    p6 = gmsh.model.geo.addPoint(l, 0, l, ms)
    p7 = gmsh.model.geo.addPoint(l, l, l, ms)
    p8 = gmsh.model.geo.addPoint(0, l, l, ms)

    # Create lines between the points to form the cube edges
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    l5 = gmsh.model.geo.addLine(p1, p5)
    l6 = gmsh.model.geo.addLine(p2, p6)
    l7 = gmsh.model.geo.addLine(p3, p7)
    l8 = gmsh.model.geo.addLine(p4, p8)

    l9 = gmsh.model.geo.addLine(p5, p6)
    l10 = gmsh.model.geo.addLine(p6, p7)
    l11 = gmsh.model.geo.addLine(p7, p8)
    l12 = gmsh.model.geo.addLine(p8, p5)

    # Define surfaces of the cube
    # Bottom face
    bottom_face = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    gmsh.model.geo.addPlaneSurface([bottom_face])

    # Top face
    top_face = gmsh.model.geo.addCurveLoop([l9, l10, l11, l12])
    gmsh.model.geo.addPlaneSurface([top_face])

    # Four side faces
    face1 = gmsh.model.geo.addCurveLoop([l1, l6, -l9, -l5])
    gmsh.model.geo.addPlaneSurface([face1])

    face2 = gmsh.model.geo.addCurveLoop([l2, l7, -l10, -l6])
    gmsh.model.geo.addPlaneSurface([face2])

    face3 = gmsh.model.geo.addCurveLoop([l3, l8, -l11, -l7])
    gmsh.model.geo.addPlaneSurface([face3])

    face4 = gmsh.model.geo.addCurveLoop([l4, l5, -l12, -l8])
    gmsh.model.geo.addPlaneSurface([face4])

    # Create a volume by joining the surfaces
    volume_loop = gmsh.model.geo.addSurfaceLoop([1, 2, 3, 4, 5, 6])
    gmsh.model.geo.addVolume([volume_loop])

    # Synchronize the model
    gmsh.model.geo.synchronize()

    # Generate the 3D mesh
    gmsh.model.mesh.generate(3)

    # Save the mesh to a file
    gmsh.write(mesh_fname)

    if visualize_mesh:
        gmsh.fltk.run()

    # Finalize the gmsh API
    gmsh.finalize()

def extract_mesh_info(mesh_fname):
    # Extract the nodes that make up each tetrahedron
    # Load the mesh from the saved .msh file
    mesh = meshio.read(mesh_fname)

    # Get the node coordinates, where the rows coorespond to node IDs
    # and the columns correspond to the x, y, and z coordinates of a node
    nodes = mesh.points  

    # Get the tetrahedral elements, i.e., which nodes make a tetrahedron
    tetrahedrons = np.sort(mesh.cells_dict.get("tetra"), axis=1)

    return (nodes, tetrahedrons)

# Function to extract the faces of a tetrahedron
def extract_faces(tet):
    # Each tet has 4 faces, formed by taking 3 nodes at a time out of 4
    faces = [
        sorted([tet[0], tet[1], tet[2]]),
        sorted([tet[0], tet[1], tet[3]]),
        sorted([tet[0], tet[2], tet[3]]),
        sorted([tet[1], tet[2], tet[3]])
    ]
    return faces

# Extract the unique faces of the tetrahedron
def extract_unique_faces(tetrahedrons):
    # Extract the faces for each tet
    all_faces = []
    for tet in tetrahedrons:
        faces = extract_faces(tet)
        all_faces.extend(faces)

    all_faces = np.array(all_faces)

    # Remove duplicate faces 
    unique_faces = set(tuple(face) for face in all_faces)
    unique_faces = np.array(list(unique_faces))

    return (all_faces, unique_faces)

# Map each tetrahedrons to its faces and vice versa
def tets_to_from_faces(tetrahedrons):
    # Create a dictionary to map each tetrahedron to its faces
    tets_to_faces = {}

    # Create a dictionary to map each face to the tets that have said face
    faces_to_tets = defaultdict(list)

    # Iterate over tets and build the mappings
    for i, tet in enumerate(tetrahedrons):
        # print(f"Tet info: {i}, {tet}")
        # Extract faces of current tetrahedron 
        faces = extract_faces(tet)

        # Map the current tet to its faces
        tets_to_faces[i] = faces

        # Map each face to the current tet
        for face in faces:
            faces_to_tets[tuple(sorted(face))].append(i)

    # For boundary tets, i.e., the tets that have a face not shared 
    # with other tets, append a "negative ID" that corresponds to a BC
    # for that tet
    for face, tets in faces_to_tets.items():
        if len(tets) == 1:
            faces_to_tets[face].append(-1)

    return tets_to_faces, faces_to_tets

def find_tet_neighbors(tets_to_faces, faces_to_tets):
    tets_to_neighbors = defaultdict(set)

    for tet_idx, faces in tets_to_faces.items():
        for face in faces:
            # Sort face to match keys in faces_to_tets
            sorted_face = tuple(sorted(face))

            # Get all tets that share this face
            tets_sharing_face = faces_to_tets[sorted_face]

            # If there are two tets sharing the same face, find the other one (neighbor)
            if len(tets_sharing_face) == 2:
                # The neighbor is the other tet, not the current tet (tet_idx)
                neighbor = tets_sharing_face[0] if tets_sharing_face[1] == tet_idx else tets_sharing_face[1]
                tets_to_neighbors[tet_idx].add(neighbor)
            # If there's only one neighbor, then it's a boundary face, so no neighboring tet
        
    # Convert sets to lists
    tets_to_neighbors = {k: list(v) for k, v in tets_to_neighbors.items()}

    return tets_to_neighbors

# Separate boundary faces from internal faces
def find_boundary_and_internal_faces(faces_to_tets):
    # Boundary faces will have only one tet corresponding to it
    boundary_faces = {}

    # Internal faces will have 2 tets corresponding to it
    internal_faces = {}

    for face, tets in faces_to_tets.items():
        if -1 in tets:
            boundary_faces[face] = tets[0]
        else:
            internal_faces[face] = tets

    return boundary_faces, internal_faces

def create_dirs(n_polar=10, n_azi=10):
    # Polar angles (Gauss-Legendre quadrature)
    mus, mu_weights = np.polynomial.legendre.leggauss(n_polar)
    sqrt_1_minus_mu2 = np.sqrt(1 - mus**2)

    # Azimuthal angles (equispaced)
    azis = np.linspace(0, 2 * np.pi, num=n_azi, endpoint=False) + np.pi / n_azi
    azi_weight = 2 * np.pi / n_azi  # Uniform weight for azimuthal angles

    # Compute the Cartesian components of the direction vectors
    cos_azi = np.cos(azis)  # Shape: (n_azi,)
    sin_azi = np.sin(azis)  # Shape: (n_azi,)
    x = sqrt_1_minus_mu2[:, None] * cos_azi  # Shape: (n_polar, n_azi)
    y = sqrt_1_minus_mu2[:, None] * sin_azi  # Shape: (n_polar, n_azi)
    z = mus[:, None] * np.ones_like(azis)    # Shape: (n_polar, n_azi)

    # Flatten the direction vectors into a single array
    dir_vecs = np.column_stack((x.ravel(), y.ravel(), z.ravel()))  # Shape: (n_polar * n_azi, 3)

    # Expand mu_weights to match azimuthal angles
    # Each polar weight is repeated n_azi times, then multiplied by azi_weight
    weights = np.repeat(mu_weights, n_azi) * azi_weight  # Shape: (n_polar * n_azi,)

    # Normalize weights to sum to 4*pi
    weights *= (4 * np.pi) / np.sum(weights)

    return dir_vecs, weights

# Compute a normal vector for the face of a tetrahedron,
# which are defined by the nodes that make up the face
def compute_normal(nodes, tet_face):
    # Extract the coordinates of the face nodes
    node1 = nodes[tet_face[0]]
    node2 = nodes[tet_face[1]]
    node3 = nodes[tet_face[2]]

    # Get two edges of the face
    edge1 = node2 - node1
    edge2 = node3 - node1

    # Compute the cross product of the two edges, which is normal
    # to the face by construction
    normal = np.cross(edge1, edge2)
    normal /= np.linalg.norm(normal)

    return normal

# Compute the outward normal to a face defined by 
# (node1, node2, node3)
# node4 is a node of the tet that doesn't make up the above face
def compute_outward_normal(nodes, tet_face, node4):
    normal = compute_normal(nodes, tet_face)

    # Edge that is in the direction of the outward normal
    # (node1 can be replaced with either node2 or node3)
    edge = nodes[tet_face[0]] - nodes[node4]

    if np.dot(normal, edge) < 0.0:
        normal = -normal
    
    return normal

# Compute the outward facing normals for each tetrahedron 
def compute_tet_normals(nodes, tetrahedrons, tets_to_faces):
    tets_normals = defaultdict(list)

    for tet_idx, tet in enumerate(tetrahedrons):
        faces = tets_to_faces[tet_idx]
        
        for face in faces:
            # Get the tetrahedron node that doesn't make up the current face
            node4 = set(tet) - set(face); node4 = node4.pop()

            ow_normal = compute_outward_normal(nodes, face, node4)
            tets_normals[tet_idx].append(ow_normal)

    return tets_normals

 # Loop over the neighboring tets of the given tet
# to reduce their dependencies as the given tet is already solved,
# and transmit information to its neighbors
def reduce_deps(tet_idx, nbr_deps_per_tet, deps_per_tet, \
                solve_buffer, sweep_order):
    for nbr_idx in nbr_deps_per_tet[tet_idx]:
        # Add directed edge from the current tet to the neighboring tet,
        # but do so only if the edge is not already in the task graph
        if not sweep_order.has_edge(tet_idx, nbr_idx):
            sweep_order.add_edge(tet_idx, nbr_idx)

        # Reduce the number of dependencies of neighboring tet
        # by 1 since one of its faces has received information from the same
        # edge of the current tet
        deps_per_tet[nbr_idx] -= 1

        # If the neighboring tet has 0 dependencies, add it to the buffer
        # of tets that are ready to solve
        if deps_per_tet[nbr_idx] == 0:
            solve_buffer.append(nbr_idx)

def compute_sweep_order(dir_vec, tetrahedrons, tets_to_faces, tets_normals,
                         boundary_faces):

    # Map to store the number of dependecies for each tet
    # Each tet starts off with 4 dependencies, equal to the number of faces in a tet
    deps_per_tet = {tet_idx: 4 for tet_idx in np.arange(tetrahedrons.shape[0])}
    
    # Map to store, for a given tet, the neighboring tets that require information from 
    # the tet's outflow faces
    nbr_deps_per_tet = {i: [] for i in np.arange(tetrahedrons.shape[0])}

    # Loop over each tet and determine the number of dependencies we can reduce:
    # (1) if the tet is a boundary tet, and has boundary faces that are inflow
    #     faces relative to the current direction vector, reduce the number of deps by however
    #     many such boundary faces
    # (2) for every tet, if the tet has outflow edges, reduce the number of deps
    #     by however many such faces
    for tet_idx, tet in enumerate(tetrahedrons):
        faces   = tets_to_faces[tet_idx]
        normals = tets_normals[tet_idx]
        for face_idx, face_normal in enumerate(normals):
            face = faces[face_idx]
            
            dp   = np.dot(dir_vec, face_normal)

            # If the direction vector is parallel to the outward normal,
            # need to differeniate if outflow or inflow face
            # This can be done by finding a direction vector from
            # the fourth vertex to one of the face vertices, and checking
            # if the dot product between the 
            # if dp == 0.0:
            #     print("neither")

            # If outflow face, reduce # of dependencies for tet by 1
            if dp > 0.0:
                deps_per_tet[tet_idx] -= 1
            
            # If boundary face and is an inflow face (relative to dir),
            # reduce the tet's dependencies by 1 since this is a incident
            # face relative to the direction
            if (tuple(face) in boundary_faces) and dp < 0.0:
                deps_per_tet[tet_idx] -= 1

    # Initialize a task-directed graph (or DAG) to determine the order
    # in which we should solve over the tets, for the current direction,
    # using the flow of information in and out of the faces
    sweep_order = nx.DiGraph()

    # Add dependencies from a tet's outflow faces to its neighboring tets'
    # inflow faces
    for tet_idx, tet in enumerate(tetrahedrons):
        faces   = tets_to_faces[tet_idx]
        normals = tets_normals[tet_idx]

        for face_idx, face_normal in enumerate(normals):
            face = faces[face_idx]

            # Find neighboring tets sharing this face
            nbr_idxs = []
            for nbr_idx in np.arange(tetrahedrons.shape[0]):
                if face in tets_to_faces[nbr_idx] and (nbr_idx != tet_idx):
                    nbr_idxs.append(nbr_idx)
            
            for nbr_idx in nbr_idxs:
                # Check if the current face of the current tet is an outflow
                # face
                # # If so, then the same face for neighboring tets is an inflow face
                dp = np.dot(dir_vec, face_normal)
                # if dp >= 0.0: # Check for outflow edge
                #     sweep_order.add_edge(tet_idx, nbr_idx)
                #     nbr_deps_per_tet[tet_idx].append(nbr_idx)
                if dp > 0.0 and (not sweep_order.has_edge(tet_idx, nbr_idx)):
                    sweep_order.add_edge(tet_idx, nbr_idx)
                    nbr_deps_per_tet[tet_idx].append(nbr_idx)

    # Buffer of tets that are ready to solve (tets that have zero dependencies)
    solve_buffer = deque([tet_idx for tet_idx, deps in deps_per_tet.items() if deps == 0])

    # Determine solve order of the tets for the current direction
    solve_order = []

    while solve_buffer:
        # Pop the next tet that is ready to solve
        tet_idx = solve_buffer.popleft()
        solve_order.append(tet_idx)
        reduce_deps(tet_idx, nbr_deps_per_tet, deps_per_tet, \
                    solve_buffer, sweep_order)
        
    # # Perform a topological sort on the task graph
    # top_order = list(nx.topological_sort(sweep_order))
    # faces_list = list(sweep_order.edges())

    # plt.figure(figsize=(8, 6))

    # # Draw nodes, edges, and labels
    # pos = nx.spring_layout(sweep_order)
    # nx.draw(sweep_order, pos, with_labels=True, node_color="lightblue", \
    #         arrows=True, node_size=500, font_size=10, font_weight="bold")
    
    # # Draw edge labels to show deps.
    # edge_labels = {(u, v): f'{u} -> {v}' for u,v in faces_list}
    # nx.draw_networkx_edge_labels(sweep_order, pos, edge_labels=edge_labels, font_color='red', font_size=5)
    # plt.show()
    
    return solve_order, sweep_order

def compute_levels(dag):
    """
    Computes the levels of nodes in a Directed Acyclic Graph (DAG).
    
    :param dag: A NetworkX DiGraph representing the DAG.
    :return: A dictionary mapping each node to its corresponding level.
    """
    # Step 1: Perform a topological sort to ensure correct processing order
    topo_sorted_nodes = list(nx.topological_sort(dag))
    
    # Step 2: Initialize a dictionary to hold the level of each node
    levels = {node: 0 for node in topo_sorted_nodes}
    
    # Step 3: Compute levels for each node in topological order
    for node in topo_sorted_nodes:
        # For each predecessor, the level is one more than the max level of its predecessors
        predecessors = list(dag.predecessors(node))
        if predecessors:
            levels[node] = max(levels[pred] for pred in predecessors) + 1

    return levels

def plot_tetrahedral_mesh(nodes, tets_to_faces, tets_to_levels, dir_vec,
                          pause_time=1.5):
    # Enable interactive mode
    plt.ion()

    # Set up the figure and 3D axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define a colormap for levels
    num_levels = max(tets_to_levels.values()) + 1
    colormap   = cm.get_cmap('tab10', num_levels)

    # Calculate mesh bounds
    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    z_min, z_max = np.min(nodes[:, 2]), np.max(nodes[:, 2])

    # Define the vector origin off to the side of the mesh
    vector_origin = np.array([x_max + 0.5, y_max + 0.5, z_max + 0.5])  # adjust the offset as needed

    # Add the direction vector
    ax.quiver(
        vector_origin[0], vector_origin[1], vector_origin[2],  # starting point of vector
        dir_vec[0], dir_vec[1], dir_vec[2],  # direction vector
        color="red", linewidth=2, arrow_length_ratio=0.1
    )

    # Set the plot limits and aspect ratio
    ax.set_aspect('auto')
    ax.set_xlim(x_min - 1, x_max + 2)  # Extend limits to fit the vector
    ax.set_ylim(y_min - 1, y_max + 2)
    ax.set_zlim(z_min - 1, z_max + 2)

    # Track plotted tetrahedrons to allow adding/removing as needed
    plotted_tetrahedrons = []

    # Start at level 0
    level = 0

    while 0 <= level < num_levels:
        # Save the current view limits and angle
        xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        elev, azim = ax.elev, ax.azim

        # Plot tetrahedrons at the current level if advancing forward
        for tet_index, tet_level in tets_to_levels.items():
            if tet_level == level:
                faces = tets_to_faces[tet_index]
                color = colormap(level / num_levels) # Map level to color
                face_coords = [nodes[face] for face in faces]
                poly = Poly3DCollection(face_coords, color=color, alpha=0.9, edgecolor="k")
                plotted_tetrahedrons.append((level, poly))  # Track the added poly for removal if needed
                ax.add_collection3d(poly)
        
        # Restore the saved limits and view angle
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.view_init(elev=elev, azim=azim)

        # Pause to display the current level before moving to the next
        plt.draw()
        plt.pause(pause_time)

        # Input control for navigation
        print("Options: 'Enter' to go forward, 'b' to go backward, 'r' to restart.")
        input_choice = input().strip().lower()

        if input_choice == 'b': # Go back a level
            # Remove tetrahedrons from the current level
            for lev, poly in plotted_tetrahedrons:
                if lev == level:
                    poly.remove()
            plotted_tetrahedrons = [(lev, poly) for lev, poly in plotted_tetrahedrons if lev != level]
            level -= 1
        elif input_choice == 'r': # Restart
            print("Restarting ...")
            for _, poly in plotted_tetrahedrons:
                poly.remove()
            plotted_tetrahedrons.clear()
            ax.quiver(
                vector_origin[0], vector_origin[1], vector_origin[2],  # starting point of vector
                dir_vec[0], dir_vec[1], dir_vec[2],  # direction vector
                color="red", linewidth=2, arrow_length_ratio=0.1
            )
            level = 0 # Reset to start over again from level 0
        else: # Advance forward
            level += 1
        

    # Keep the plot open after all levels are displayed
    plt.ioff()
    plt.show()

def compute_total_frames(tetrahedrons, tets_to_faces, tets_normals,
                         boundary_faces, dir_vecs):
    total_frames = 0

    for dir_vec in dir_vecs:
        # Compute sweep order for the current direction vector
        _, sweep_order = compute_sweep_order(dir_vec, tetrahedrons, tets_to_faces,
                                            tets_normals, boundary_faces)
        
        # Mapping from tetrahedron index to its sweep order level
        tets_to_levels = compute_levels(sweep_order)

        # Add the max number of levels in the current sweep order
        # to the total_frames variable
        # We sum over all the max number of levels for each sweep order to ensure
        # that when we plot the sweep animation for a given direction vector, there
        # are enough animation frames to display the entire sweep order

        total_frames += max(tets_to_levels.values()) + 1

    return total_frames

def create_combined_animation(nodes, tetrahedrons, tets_to_faces, tets_normals,
                              boundary_faces, direction_vectors, total_frames,
                              output_file="multi_direction_sweep.mp4", save_as_gif=False):
    """
    Create an animation showing sweep order for multiple direction vectors.
    
    Parameters:
    - nodes: Array of node coordinates.
    - tets_to_faces: Dictionary mapping tetrahedron indices to face node indices.
    - direction_vectors: List of direction vectors for the sweeps.
    - output_file: Name of the output file (default: MP4 format).
    - save_as_gif: Whether to save the animation as a GIF instead of MP4.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Set up zoomed view
    x_min, x_max = np.min(nodes[:, 0]), np.max(nodes[:, 0])
    y_min, y_max = np.min(nodes[:, 1]), np.max(nodes[:, 1])
    z_min, z_max = np.min(nodes[:, 2]), np.max(nodes[:, 2])
    margin = 0.5
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)

    plot_objects = []
    levels_cache = {}  # Cache to store tetrahedron levels for each direction vector
    frame_to_direction = []  # Map each frame to a direction vector and level

    # Precompute frame_to_direction for all directions and levels
    for current_vector in direction_vectors:
        if current_vector not in levels_cache:
            _, dag = compute_sweep_order(current_vector, tetrahedrons, tets_to_faces,
                                          tets_normals, boundary_faces)  # Generate the DAG
            levels_cache[current_vector] = compute_levels(dag)  # Compute levels
        
        levels = levels_cache[current_vector]
        num_levels = max(levels.values()) + 1
        frame_to_direction.extend([(current_vector, level) for level in range(num_levels)])

    def update(frame):
        nonlocal plot_objects, frame_to_direction

        # Get direction vector and level for the current frame
        current_vector, current_level = frame_to_direction[frame]

        # Remove previous plotted objects
        while plot_objects:
            plot_objects.pop().remove()

        # Update title with the current direction vector
        ax.set_title(f"Direction: [{current_vector[0]:.2f}, {current_vector[1]:.2f}, {current_vector[2]:.2f}]", fontsize=12)

        # Plot tetrahedrons up to the current level
        levels = levels_cache[current_vector]
        colormap = cm.get_cmap("tab10", len(set(levels.values())))
        for tet_index, tet_level in levels.items():
            if tet_level <= current_level:
                faces = tets_to_faces[tet_index]
                face_coords = [nodes[face] for face in faces]
                poly = Poly3DCollection(face_coords, color=colormap(tet_level), alpha=0.6, edgecolor="k")
                ax.add_collection3d(poly)
                plot_objects.append(poly)

        # Plot the direction vector
        vector_origin = np.array([x_max + margin, y_max + margin, z_max + margin])
        vector_plot = ax.quiver(
            vector_origin[0], vector_origin[1], vector_origin[2],  # Starting point
            current_vector[0], current_vector[1], current_vector[2],  # Direction
            color="red", linewidth=2, arrow_length_ratio=0.1
        )
        plot_objects.append(vector_plot)

    # Total number of frames based on the precomputed frame_to_direction
    total_frames = len(frame_to_direction)

    # Create the FuncAnimation with a range of frames
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000, blit=False)

    # Save the combined animation
    ani.save(output_file, writer="ffmpeg", fps=10)

    if save_as_gif:
        ani.save(output_file.replace(".mp4", ".gif"), writer="pillow", fps=10)

    plt.close(fig)

def compute_jacobian(tet_nodes, nodes):
    # Jacobian matrix of transformation from physical tet to reference tet
    # Each column of the matrix is of the form 
    # [x_i - x_0, y_i - y_0, z_i - z_0]
    jac = np.zeros((3, 3))
    for idx in np.arange(0, 3):
        jac[:, idx] = nodes[tet_nodes[idx + 1], :] - nodes[tet_nodes[0], :]

    # The absolute value of the determinant of the Jacobian is
    # the volume scaling factor from the reference tet to the physical tet
    det_jac     = np.linalg.det(jac)
    abs_det_jac = np.abs(det_jac)

    return jac, det_jac, abs_det_jac

def compute_outflow_jacobian(face_coords):
    j1 = face_coords[1, :] - face_coords[0, :]
    j2 = face_coords[2, :] - face_coords[0, :]
    jacobian = np.linalg.norm(np.cross(j1, j2))
    return jacobian

def outflow_contrib(face_nodes, tet_nodes):
    m_2d = np.full((4, 4), 1/24.0)
    np.fill_diagonal(m_2d, 1/12.0)

    non_face_node = np.setdiff1d(tet_nodes, face_nodes)
    non_face_node_idx = np.nonzero(np.isin(tet_nodes, non_face_node))[0]

    m_2d[non_face_node_idx, :] = 0.0
    m_2d[:, non_face_node_idx] = 0.0

    return m_2d

def total_interaction_contrib():
    tot_int_mat = np.full((4, 4), 1.0)
    np.fill_diagonal(tot_int_mat, 2.0)
    tot_int_mat *= (1/120.0)

    return tot_int_mat

def streaming_contrib(d, tet_jacobian, tet_jacobian_det):
    # Construct streaming linear system

    # One can construct the Jacobian explicitly as follows,
    # but I need to double check the math on this 
    # inv_jac_t = np.zeros((3, 3))
    # inv_jac_t[0, :] = np.cross(jac[:, 1], jac[:, 2])
    # inv_jac_t[1, :] = np.cross(jac[:, 2], jac[:, 0])
    # inv_jac_t[2, :] = np.cross(jac[:, 0], jac[:, 1])
    # inv_jac_t /= det_jac

    # For the time being, compute the inverse of the transpose
    # of the Jacobian using NumPy
    # This is computationally expensive, and I'll put in the 
    # explicit expression for this inverse later
    streaming_mat = np.ones((4, 4))

    inv_jac_T = np.linalg.inv(tet_jacobian.T)

    grad_b1 = np.array([-1.0, -1.0, -1.0]).reshape((3, ))
    grad_b2 = np.array([1.0, 0.0, 0.0]).reshape((3, ))
    grad_b3 = np.array([0.0, 1.0, 0.0]).reshape((3, ))
    grad_b4 = np.array([0.0, 0.0, 1.0]).reshape((3, ))

    streaming_mat[0, :] *= (np.array(d) @ inv_jac_T @ grad_b1)
    streaming_mat[1, :] *= (np.array(d) @ inv_jac_T @ grad_b2)
    streaming_mat[2, :] *= (np.array(d) @ inv_jac_T @ grad_b3)
    streaming_mat[3, :] *= (np.array(d) @ inv_jac_T @ grad_b4)
    
    streaming_mat *= -(tet_jacobian_det/24.0)

    return streaming_mat

def main():
    # Create the mesh
    # l = 1.0
    # ms = 0.9
    mesh_fname = "cube.msh"
    # visualize_mesh = False
    # create_cubic_mesh(l, ms, mesh_fname, visualize_mesh=visualize_mesh)

    # Extract the mesh info
    nodes, tetrahedrons = extract_mesh_info(mesh_fname)

    # all_faces, unique_faces = extract_unique_faces(tetrahedrons)

    #------------------------------------
    # Useful mappings and data structures
    #------------------------------------

    # Mappings that:
    # map each tet to its face 
    # map each face to the tets that share that face
    tets_to_faces, faces_to_tets = tets_to_from_faces(tetrahedrons)

    # Extract coordinates for each tetrahedron's vertices
    # Has shape (N, 4, 3), where N is the number of tets
    tetrahedrons_coords = nodes[tetrahedrons]

    # The face indices (local node indices in a tet) for a tet
    face_indices = np.array([
        [0, 1, 2],  # Face formed by nodes 0,1,2 of the tetrahedron
        [0, 1, 3],  # Face formed by nodes 0,1,3
        [0, 2, 3],  # Face formed by nodes 0,2,3
        [1, 2, 3]   # Face formed by nodes 1,2,3
    ])

    # (N, 4, 3, 3) array, where 
    # ith row contains the coordinates for each face that make up ith tet
    faces_coords = tetrahedrons_coords[:, face_indices, :]

    # Edge vectors 
    edge1 = faces_coords[:,:,1,:] - faces_coords[:,:,0,:]  # shape: (N,4,3)
    edge2 = faces_coords[:,:,2,:] - faces_coords[:,:,0,:]  # shape: (N,4,3)

    # Cross product to get unnormalized normals
    face_normals = np.cross(edge1, edge2)  # shape: (N,4,3)
    
    norms = np.linalg.norm(face_normals, axis=2, keepdims=True)  # shape: (N,4,1)
    face_normals = face_normals / norms

    # For each face, there is a tet node that doesn't make up the face
    # These are the `opposite` nodes corresponding to the way that the face nodes
    # have been ordered in face_indices
    opposite_nodes = np.array([3, 2, 1, 0])

    # Shape: (N,4,3)
    opposite_coords = tetrahedrons_coords[np.arange(len(tetrahedrons_coords))[:,None], 
                                          opposite_nodes, :]  
    # Vector from a face vertex to opposite vertex (N,4,3)
    to_opposite = opposite_coords - faces_coords[:,:,0,:]  

    # If dot product > 0, normal is pointing inward, so negate it
    dot_products = np.sum(face_normals * to_opposite, axis=2)  # shape: (N,4)
    flip_mask = (dot_products > 0.0)
    face_normals[flip_mask] = -face_normals[flip_mask]

    boundary_faces, _ = find_boundary_and_internal_faces(faces_to_tets)

    # NOTE: This is redunant; refactor code to make use of already computed
    # `face_normals`
    # Compute the outward facing normals for the tetrahedrons
    tets_normals = compute_tet_normals(nodes, tetrahedrons, tets_to_faces)

    # Create discrete direction vectors
    n_polar = 20
    n_azi   = 20

    dir_vecs, glc_weights = create_dirs(n_polar=n_polar, n_azi=n_azi)
    
    """
    # Total number of animation frames to create full sweep animation
    # for all the direction vectors
    total_frames = compute_total_frames(tetrahedrons, tets_to_faces, tets_normals,
                                        boundary_faces, dir_vecs)
    
    # Create the combined animation (as MP4 and/or GIF)
    create_combined_animation(nodes, tetrahedrons, tets_to_faces, tets_normals,
                              boundary_faces, dir_vecs, total_frames, output_file="multi_direction_sweep.mp4",
                              save_as_gif=True)
    """

    # NOTE: The below plotting code should be used
    # Plot the overall tetrahedral mesh and a given direction vector
    # plot_tetrahedral_mesh(nodes, tets_to_faces, tets_to_levels, dir_vec)

    # Isotropic, distributed volumetric source
    # For time being, set it constant for each cell
    q = 3.0/(4*np.pi)

    # Total interaction cross-section
    # For time being, set it constant for each cell
    sigma_t = 1.0

    # eps = 1e-10

    # Array to store boundary conditions for boundary faces
    # For an isotropic, distributed volumetric source and constant total interaction cross section (same for every tet),
    # the angular and scalar fluxes will be constant and equal to q/(sigma_t * 4 * pi) [q/sigma_t in this case b/c of division by 4*pi
    # in q]
    incident_boundary_fluxes = q/sigma_t
    boundary_fluxes = np.zeros((tetrahedrons.shape[0], 4, 3))
    for tet in np.arange(tetrahedrons.shape[0]):
        for face_idx, face_nodes in enumerate(tets_to_faces[tet]):
            # If face is a boundary face, impose BCs
            if tuple(face_nodes) in boundary_faces:
                boundary_fluxes[tet, face_idx] = incident_boundary_fluxes
    
    # Array to store angular fluxes 
    # ith row corresponds to ith cell, and jth row corresponds to jth angular flux in ith cell)
    angular_fluxes = np.zeros((tetrahedrons.shape[0], 4))

    # Array to store scalar fluxes
    # ith row corresponds to ith cell
    scalar_fluxes = np.zeros((tetrahedrons.shape[0], ))

    # Create LHS matrix and RHS vector for solving over tets
    # (will be zeroed out after solving a tet)
    A = np.zeros((4, 4))
    b = np.zeros((4, ))

    # 2nd attempt at transport solve, using the following looping order:
    # Loop for directions and compute sweep order
    for d_idx, d in enumerate(dir_vecs):
        solve_order, sweep_order = compute_sweep_order(d, tetrahedrons, tets_to_faces,
                                                       tets_normals, boundary_faces)
        
        # # Mapping from sweep order level to tetrahedron index
        # tets_to_levels = compute_levels(sweep_order)
        # levels_to_tets = defaultdict(list)
        # for tet, level in tets_to_levels.items():
        #     levels_to_tets[level].append(tet)

        # max_levels = np.max(list(tets_to_levels.values()))

        # Loop through each tet at current sweep order level
        for tet in solve_order:
            tet_nodes = tetrahedrons[tet]

            # Compute Jacobian J of transformation from tet to reference
            # 3D tet, as well as abs(|J|)
            tet_jacobian, _, tet_jacobian_det = compute_jacobian(tet_nodes, nodes)

            # Compute total interaction contribution
            tot_int_mat = total_interaction_contrib()
            A += (sigma_t * tet_jacobian_det * tot_int_mat)

            # Compute streaming contribution
            streaming_mat = streaming_contrib(d, tet_jacobian,
                                                tet_jacobian_det)
            A += streaming_mat

            # Loop through each face of the current tet to add outflow 
            # and inflow contributions
            for face_idx, face_nodes in enumerate(tets_to_faces[tet]):
                # Face coordinates
                face_coords = faces_coords[tet, face_idx]
                
                # Outward pointing normal for face
                face_normal = face_normals[tet, face_idx]

                # Compute Jacobian of transformation from 
                # face to 2D reference triangle
                face_jacobian_det = compute_outflow_jacobian(face_coords)

                alpha = np.dot(d, face_normal)

                # Compute outflow/inflow contribution matrix
                m_2d = outflow_contrib(face_nodes, tet_nodes)

                # Outflow face
                if alpha > 0.0:
                    A += (alpha * face_jacobian_det * m_2d)
                # Inflow face
                else:
                    # Find the relative ordering of the nodes on this face for the current tet
                    curr_tet_node_rel_order = face_indices[face_idx]

                    # Compute inflow contribution (requires upwind flux info)
                    # Find neighboring tet that shares this face
                    upwind_tet = np.array(faces_to_tets[tuple(face_nodes)])
                    upwind_tet = upwind_tet[upwind_tet != tet][0]

                    permuted_angular_fluxes = np.zeros((4, ))
                    # Check if the face is a boundary face
                    # If it is a boundary face, then extract the incident flux info for this face
                    if upwind_tet == -1:
                        upwind_angular_fluxes = boundary_fluxes[tet, face_idx]
                        permuted_angular_fluxes[curr_tet_node_rel_order] = upwind_angular_fluxes

                    # Otherwise, grab the upwind tet angular fluxes for this face
                    else:
                        # Find the relative ordering of this face in the upwind tet faces
                        # This determines how we permute the upwind angular fluxes for this face
                        # to match the relative ordering of nodes for the same face on the current
                        # tet
                        upwind_tet_face_idx       = tets_to_faces[upwind_tet].index(face_nodes)
                        upwind_tet_node_rel_order = face_indices[upwind_tet_face_idx]
                        upwind_angular_fluxes     = angular_fluxes[upwind_tet]

                        permuted_angular_fluxes[curr_tet_node_rel_order] = upwind_angular_fluxes[upwind_tet_node_rel_order]

                    b -= (alpha * face_jacobian_det * (m_2d @ permuted_angular_fluxes))

            # Add volumetric source contribution
            b += (q * tet_jacobian_det * np.full((4, ), 1/24.0))

            # Solve for tet angular fluxes for this direction
            angular_fluxes[tet] = np.linalg.solve(A, b)

            # Zero out LHS matrix and RHS vector for next tet solve
            A[:, :] = 0.0; b[:] = 0.0

        # print(f"Angular fluxes = {angular_fluxes}")
        # Update angular flux contributions to scalar fluxes for this direction
        # Angular fluxes are volume averaged (for each tet, we take an equal contribution from the 4 angular fluxes)
        scalar_fluxes += glc_weights[d_idx] * 0.25 * np.sum(angular_fluxes, axis=1)

        # Zero out angular fluxes for next direction
        angular_fluxes[:, :] = 0.0
    
    # compare_scalar_fluxes = np.isclose(scalar_fluxes - 4*np.pi*q/sigma_t, 
    #                                     np.zeros_like(scalar_fluxes))
    # print(compare_scalar_fluxes)

    ### Visualize the scalar fluxes using VTK
    from pyvista import CellType

    # Step 1: Define the connectivity for tetrahedral cells
    # Each cell has a "connectivity" consisting of the number of points (4 for tetrahedrons) + node indices
    num_cells = len(tetrahedrons)
    cell_types = np.full(num_cells, CellType.TETRA, dtype=np.uint8)

    # Flatten tetrahedral connectivity and prepend the number of points per cell
    connectivity = np.hstack([[4] + list(cell) for cell in tetrahedrons])

    # Step 2: Create a PyVista UnstructuredGrid
    grid = pv.UnstructuredGrid(connectivity, cell_types, nodes)

    # Step 3: Add scalar flux as cell data
    grid.cell_data['Scalar Flux'] = scalar_fluxes

    # Step 4: Visualize the scalar flux
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars='Scalar Flux', cmap='viridis', show_edges=True)
    plotter.add_scalar_bar(title="Scalar Flux")
    plotter.show()

main()