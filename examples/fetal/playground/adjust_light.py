from aitviewer.scene.light import Light
from aitviewer.viewer import Viewer

# Create a viewer object
viewer = Viewer()

# Add a light source to the viewer
light = Light()

# Set the light position (e.g., X, Y, Z coordinates)
print("light.position: {}".format(light.position))
light.position = [1.0, 2.0, 3.0]

# Set the light direction (e.g., as a unit vector)
print("light.direction: {}".format(light.direction))
light.direction = [0.0, -1.0, 0.0]

# Add the light to the viewer
viewer.scene.add_light(light)

# Start the viewer
viewer.run()
