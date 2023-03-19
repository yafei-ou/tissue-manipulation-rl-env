import Sofa.Core

def add_scene_header(
        root: Sofa.Core.Node,
        gravity: list = [0, -9.81, 0],
        dt: float = 0.01,
        ):
    root.dt.value = dt
    root.gravity.value = gravity
    root.addObject('DefaultVisualManagerLoop')
    root.addObject('VisualStyle', displayFlags='hideCollisionModels hideForceFields hideBehaviorModels showVisualModels')
    root.addObject('BackgroundSetting', color=[0, 0, 0, 0])
    root.addObject("LightManager")
    root.addObject("DirectionalLight", direction=[0,1,1])
    root.addObject("InteractiveCamera", name="camera", position=[0, 0, 35],
                            lookAt=[0,0,0], distance=150,
                            fieldOfView=45, zNear=0.63, zFar=200)
    root.addObject('FreeMotionAnimationLoop')
    root.addObject('GenericConstraintSolver', 
                    maxIterations=1000, 
                    tolerance=1e-6, 
                    printLog=0, 
                    allVerified=0
                    )
