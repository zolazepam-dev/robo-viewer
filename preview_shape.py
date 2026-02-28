import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

def plot_dodecahedron():
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi
    v = np.array([
        [ 1,  1,  1], [ 1,  1, -1], [ 1, -1,  1], [ 1, -1, -1],
        [-1,  1,  1], [-1,  1, -1], [-1, -1,  1], [-1, -1, -1],
        [ 0,  inv_phi,  phi], [ 0,  inv_phi, -phi], [ 0, -inv_phi,  phi], [ 0, -inv_phi, -phi],
        [ inv_phi,  phi,  0], [ inv_phi, -phi,  0], [-inv_phi,  phi,  0], [-inv_phi, -phi,  0],
        [ phi,  0,  inv_phi], [ phi,  0, -inv_phi], [-phi,  0,  inv_phi], [-phi,  0, -inv_phi]
    ])

    pentagons = [
        [0, 16, 2, 10, 8],   [0, 8, 4, 14, 12],   [0, 12, 1, 17, 16],
        [1, 9, 11, 3, 17],   [1, 12, 14, 5, 9],   [2, 13, 15, 6, 10],
        [2, 16, 17, 3, 13],  [3, 11, 7, 15, 13],  [4, 8, 10, 6, 18],
        [4, 18, 19, 5, 14],  [5, 19, 7, 11, 9],   [6, 15, 7, 19, 18]
    ]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    poly_faces = [[v[i] for i in p] for p in pentagons]
    ax.add_collection3d(Poly3DCollection(poly_faces, facecolors='gold', linewidths=1, edgecolors='black', alpha=.4))

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_title("Regular Dodecahedron (12 Pentagons)")
    
    plt.savefig("dodecahedron_preview.png")
    print("Saved preview to dodecahedron_preview.png")

if __name__ == "__main__":
    plot_dodecahedron()
