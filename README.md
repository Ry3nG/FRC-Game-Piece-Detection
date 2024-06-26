# FRC-Game-Piece-Detection
Game Piece Detection with Roboflow dataset and YOLO model

### Dataset Usage
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="CNJg44BjnMkpfi8O52Xv")
project = rf.workspace("frc-3707").project("2024-game-piece")
version = project.version(2)
dataset = version.download("yolov8")
