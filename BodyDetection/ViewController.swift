/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The sample app's main view controller.
*/

import UIKit
import RealityKit
import ARKit
import Combine
import CoreML
import SceneKit
import SceneKit.ModelIO

class ViewController: UIViewController, SCNSceneRendererDelegate { // ARSessionDelegate {
    // The 3D character to display.
//    let characterAnchor = AnchorEntity()
    var maleSmplModel: test_male_smpl!
    
    var smplMeshData: SMPLMetalMeshData!
    var tshirtData: ClothMetalMeshData!
    var device: MTLDevice!
    var deformer: MetalMeshDeformer!
    var newPlaneNode: SCNNode!
    var tshirtNode: SCNNode!
    
    var betaPoseMLArray = try? MLMultiArray(shape: [82], dataType: .float32)
    var zeroOffsetMLArray = try? MLMultiArray(shape: [6890, 3], dataType: .float32)
    var betaPosePointer: UnsafeMutablePointer<Float32>!
    
    var leftHandDir:Float = 1.0
    var bodyDir = 1.0
    
    var captureSession: AVCaptureSession!
    var dataOutputQueue: DispatchQueue!
    
    let outputStride = 16
    var poseNetModel: PoseNetMobileNet100S16FP16!
    var poseQueue: DispatchQueue!

    override func viewDidLoad() {
        super.viewDidLoad()
        let arView = self.view as! SCNView
        
        maleSmplModel = test_male_smpl()
        
        let faceInputStream = InputStream(url: Bundle.main.url(forResource: "faces_int32", withExtension: "bin")!)
        var faceISByteStream = Array<UInt8>(repeating: 0, count: 13776*3*4) // [13776, 3] Int32s
        faceInputStream?.open()
        faceInputStream?.read(&faceISByteStream, maxLength: 13776*3*4)
        faceInputStream?.close()
        var indexFaces = Array<Int32>(repeating: 0, count: 13776*3)
        indexFaces.withUnsafeMutableBytes {w in
            faceISByteStream.copyBytes(to: w)
        }
        
        betaPosePointer = UnsafeMutablePointer<Float32>(OpaquePointer(betaPoseMLArray!.dataPointer))
        for i in 0..<10 {
            betaPosePointer[i] = Float.random(in: -1.0..<1.0)
        }
        for i in 10..<82 {
            betaPosePointer[i] = 0.0
        }
        
        guard let tvertices = try? maleSmplModel.prediction(betas_pose_trans: betaPoseMLArray!, v_personal: zeroOffsetMLArray!) else {
            fatalError("Failed to load vertices")
        }
        
        poseNetModel = PoseNetMobileNet100S16FP16()
        poseQueue = DispatchQueue(label: "PoseNet", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem, target: nil)
       
        let scene = SCNScene()
        arView.scene = scene
        
        let light = SCNLight()
        light.color = UIColor(red: 0.2, green: 0.2, blue: 0.2, alpha: 1)
        light.type = SCNLight.LightType.omni
        let lightNode = SCNNode()
        lightNode.light = light
        lightNode.position.z = -3
        arView.pointOfView?.addChildNode(lightNode)
        
        arView.preferredFramesPerSecond = 30
        
        device = MTLCreateSystemDefaultDevice()
        deformer = MetalMeshDeformer(device: device)
        
        smplMeshData = MetalMeshDeformable.buildSMPL(device, vertices: tvertices._1246, faces: indexFaces)
        newPlaneNode = SCNNode(geometry: smplMeshData.geometry)
        newPlaneNode.castsShadow = true
        scene.rootNode.addChildNode(newPlaneNode)
        
        tshirtData = MetalMeshDeformable.buildMGM(device, folderName: "1", smplVertices: tvertices._1246)
        tshirtNode = SCNNode(geometry: tshirtData.geometry)
        tshirtNode.castsShadow = true
        scene.rootNode.addChildNode(tshirtNode)
        
        // deformer.initDirs(smplMeshData, clothMesh: tshirtData)

        self.initCamera()
        
        arView.delegate = self
        arView.allowsCameraControl = true
        arView.showsStatistics = true
        arView.backgroundColor = UIColor.lightGray
        arView.autoenablesDefaultLighting = true
        arView.isPlaying = true
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        self.startCapture()
    }
    
    func renderer(_ renderer: SCNSceneRenderer, willRenderScene scene: SCNScene, atTime time: TimeInterval) {
//        betaPosePointer[10+(16*3+2)] -= leftHandDir*0.03
//        betaPosePointer[10+(14*3+1)] -= leftHandDir*0.03
//        // betaPoseArray[10+(20*3+1)] -= Float32(leftHandDir*0.03)
//        if(betaPosePointer[10+(16*3+2)] < -1.35 || betaPosePointer[10+(16*3+2)] > 1.35)
//        {
//            leftHandDir *= -1.0
//        }
        
        // optimize: Store the betas_pose_trans MLMultiArray and just modify the memory using pointers
        
        guard let tvertices = try? maleSmplModel.prediction(betas_pose_trans: betaPoseMLArray!, v_personal: zeroOffsetMLArray!) else {
            fatalError("Failed to load vertices")
        }
        
//        guard let clothedVertices = try? maleSmplModel.prediction(betas_pose_trans: bpt, v_personal: tshirtData.offsetMLArray) else {
//            fatalError("Failed to load vertices")
//        }
        
        deformer.deformSMPL(smplMeshData, vertices: tvertices._1246)
        deformer.deformMGM(tshirtData, smplMesh: smplMeshData, vertices: tvertices._1246)
    }
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func initCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .vga640x480
        dataOutputQueue = DispatchQueue(label: "Video data", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem, target: nil)
        
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
            print("NO CAMERA")
            return
        }
        do {
            let cameraInput = try AVCaptureDeviceInput(device: camera)
            captureSession.addInput(cameraInput)
        } catch {
            print(error.localizedDescription)
        }

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: dataOutputQueue)
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]

        captureSession.addOutput(videoOutput)

        let videoConnection = videoOutput.connection(with: .video)
        videoConnection?.videoOrientation = .portrait

        videoConnection?.isVideoMirrored = true
    }
    
    func startCapture() {
        captureSession.startRunning()
    }

    func stopCapture() {
        captureSession.stopRunning()
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let croppedBuffer = resizePixelBuffer(imageBuffer, cropX: 0, cropY: 0, cropWidth: 480, cropHeight: 480, scaleWidth: 257, scaleHeight: 257)
        
//        poseQueue.async {
            guard let poseOutput = try? self.poseNetModel.prediction(image: croppedBuffer!) else {
                return
            }
            let leftShoulderPoint = getJointPosition(.leftShoulder, offsetArray: poseOutput.offsets, confidenceArray: poseOutput.heatmap)
            let leftElbowPoint = getJointPosition(.leftElbow, offsetArray: poseOutput.offsets, confidenceArray: poseOutput.heatmap)
            if leftShoulderPoint.x == -1 || leftShoulderPoint.y == -1 || leftElbowPoint.x == -1 || leftElbowPoint.y == -1 {
                // do nothing
            } else {
                let leftArmVector = leftElbowPoint - leftShoulderPoint
                print(leftArmVector)
                let zAngle = atan2f(Float(leftArmVector.y), Float(leftArmVector.x))
                print(zAngle)
                self.betaPosePointer[10+(14*3+2)] = zAngle
            }
            // get bestCellLocations for left & right shoulder
            // get offset and add to cell location
            // calculate angle
            // update betaPosePointer
//        }
    }
}

func -(lhs: CGPoint, rhs: CGPoint) -> CGPoint {
    return CGPoint(x: lhs.x-rhs.x, y: lhs.y-rhs.y)
}
