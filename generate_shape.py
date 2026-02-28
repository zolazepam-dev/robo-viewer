import numpy as np

def generate_spiky_relic(filename):
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1 / phi

    # 1. The Dodecahedron Core (20 vertices)
    core_verts = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                core_verts.append([x, y, z])
    for y in [-inv_phi, inv_phi]:
        for z in [-phi, phi]:
            core_verts.append([0, y, z])
    for x in [-inv_phi, inv_phi]:
        for y in [-phi, phi]:
            core_verts.append([x, y, 0])
    for x in [-phi, phi]:
        for z in [-inv_phi, inv_phi]:
            core_verts.append([x, 0, z])

    # 2. Add 12 MASSIVE SPIKES (One for each face)
    # Face normals for a dodecahedron are the vertices of an icosahedron
    spike_directions = [
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    
    all_vertices = list(core_verts)
    spike_tip_dist = 5.0 # Giant spikes!
    
    for direction in spike_directions:
        norm = np.linalg.norm(direction)
        tip = (np.array(direction) / norm) * spike_tip_dist
        all_vertices.append(tip.tolist())

    # Scale everything up
    all_vertices = (np.array(all_vertices) * 3).tolist()

    with open(filename, 'w') as f:
        f.write("<JoltShape type=\"SpikyRelic\">\n")
        for v in all_vertices:
            f.write(f"  <Vertex x=\"{v[0]:.4f}\" y=\"{v[1]:.4f}\" z=\"{v[2]:.4f}\" />\n")
        f.write("</JoltShape>\n")
    print(f"Generated SPIKY RELIC in {filename}")

if __name__ == "__main__":
    generate_spiky_relic("dodecahedron.xml")
