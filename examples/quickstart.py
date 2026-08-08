# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

if __name__ == "__main__":
    Viewer.window_type = "pyqt6"
    v = Viewer()
    v.scene.add(SMPLSequence.t_pose())
    v.run()
