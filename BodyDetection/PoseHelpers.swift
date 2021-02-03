//
//  PoseHelpers.swift
//  BodyDetection
//
//  Created by Vidur Satija on 14/08/20.
//  Copyright © 2020 Apple. All rights reserved.
//

import Foundation
import CoreML

enum JointName: Int, CaseIterable {
    case nose
    case leftEye
    case rightEye
    case leftEar
    case rightEar
    case leftShoulder
    case rightShoulder
    case leftElbow
    case rightElbow
    case leftWrist
    case rightWrist
    case leftHip
    case rightHip
    case leftKnee
    case rightKnee
    case leftAnkle
    case rightAnkle
}

func getJointPosition(_ j: JointName, offsetArray: MLMultiArray, confidenceArray: MLMultiArray) -> CGPoint {
    var bestCell = SIMD2(x: -1, y: -1)
    var bestConfidence: Float = 0.0
    for yIndex in 0..<offsetArray.shape[1].intValue {
        for xIndex in 0..<offsetArray.shape[2].intValue {
            let currentConfidence = confidenceArray[[j.rawValue, yIndex, xIndex] as [NSNumber]].floatValue

            // Keep track of the cell with the greatest confidence.
            if currentConfidence > bestConfidence {
                bestConfidence = currentConfidence
                bestCell.x = yIndex
                bestCell.y = xIndex
            }
        }
    }

    // Update joint.
    if bestConfidence > 0.3 {
        var ret = CGPoint(x: bestCell.x*16, y: bestCell.y*16)
        ret.x += CGFloat(offsetArray[[j.rawValue+17, bestCell.y, bestCell.x] as [NSNumber]].floatValue)
        ret.y += CGFloat(offsetArray[[j.rawValue, bestCell.y, bestCell.x] as [NSNumber]].floatValue)
        return ret;
    } else {
        return CGPoint(x: -1, y: -1)
    }
}
