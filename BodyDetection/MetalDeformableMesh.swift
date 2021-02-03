//
//  MetalMeshDeformable.swift
//  DeformableMesh
//
// Copyright (c) 2015 Lachlan Hurst
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import Foundation
import Metal
import SceneKit
import UIKit
import MetalKit
import CoreML

class SMPLMetalMeshData {
    
    var geometry:SCNGeometry
    var vertexBuffer: MTLBuffer // float3 Vx3
    var vertexCount: Int // V
    var normalBuffer: MTLBuffer // float3 Vx3
    var faceIndices: MTLBuffer // int Fx3
    var faceCount: Int // F
    var dirsBuffer: MTLBuffer // float F
    // CALCULATE NORMALS ONLY FOR CLOTH VERTICES!!!
    // How do we know which vertex belongs to the normal? The one for which the ||offset|| > 0
    init(
        geometry: SCNGeometry,
        vertexCount: Int,
        vertexBuffer: MTLBuffer,
        normalBuffer: MTLBuffer,
        faceIndices: MTLBuffer,
        faceCount: Int,
        dirsBuffer: MTLBuffer
    ) {
        self.geometry = geometry
        self.vertexCount = vertexCount
        self.vertexBuffer = vertexBuffer
        self.normalBuffer = normalBuffer
        self.faceIndices = faceIndices
        self.faceCount = faceCount
        self.dirsBuffer = dirsBuffer
    }
    
}

class ClothMetalMeshData {
    
    var geometry: SCNGeometry
    var vertexBuffer: MTLBuffer // float3 Vx3
    var vertexCount: Int // V
    var offsetMLArray: MLMultiArray // Vx3
    var vertInds: MTLBuffer // int V
    var offsetMTLBuffer: MTLBuffer // float3 Vx3
    init(
        geometry:SCNGeometry,
        vertexCount:Int,
        vertexBuffer:MTLBuffer,
        offsetMLArray: MLMultiArray,
        vertInds: MTLBuffer,
        offsets: MTLBuffer
    ) {
        self.geometry = geometry
        self.vertexCount = vertexCount
        self.vertexBuffer = vertexBuffer
        self.offsetMLArray = offsetMLArray
        self.vertInds = vertInds
        self.offsetMTLBuffer = offsets
    }
    
}

/*
Encapsulate the 'Metal stuff' within a single class to handle setup and execution
of the compute shaders.
*/
class MetalMeshDeformer {
    
    let device:MTLDevice
    
    var commandQueue: MTLCommandQueue!
    // once at start
    var pipelineStateGetDirs: MTLComputePipelineState!
    
    // every frame
    var pipelineStateInitNormals: MTLComputePipelineState!
    var pipelineStateCalcNormals: MTLComputePipelineState!
    var pipelineStateNormNormals: MTLComputePipelineState!
    
    // every frame
    var pipelineStateSlice: MTLComputePipelineState!
    
    var vertexBuffer1: MTLBuffer!
    
    init(device:MTLDevice) {
        self.device = device
        setupMetal()
    }

    func setupMetal() {
        commandQueue = device.makeCommandQueue()
        vertexBuffer1 = device.makeBuffer(length: 6890*MemoryLayout<vector_float3>.size, options: [.cpuCacheModeWriteCombined])
        
        let defaultLibrary = device.makeDefaultLibrary()
        
        let functionGetDirs = defaultLibrary!.makeFunction(name: "dirsPerFaceNormals")
        
        let functionInitNormals = defaultLibrary!.makeFunction(name: "initializeNormals")
        let functionCalcNormals = defaultLibrary!.makeFunction(name: "perFaceNormals")
        let functionNormNormals = defaultLibrary!.makeFunction(name: "perVextexNormalize")
        
        let functionSlice = defaultLibrary!.makeFunction(name: "sliceVertex")

        do {
            
            pipelineStateGetDirs = try! device.makeComputePipelineState(function: functionGetDirs!)
            
            pipelineStateInitNormals = try! device.makeComputePipelineState(function: functionInitNormals!)
            pipelineStateCalcNormals = try! device.makeComputePipelineState(function: functionCalcNormals!)
            pipelineStateNormNormals = try! device.makeComputePipelineState(function: functionNormNormals!)
            
            pipelineStateSlice = try! device.makeComputePipelineState(function: functionSlice!)
        }
    }
    
    func initDirs(_ smplMesh: SMPLMetalMeshData, clothMesh: ClothMetalMeshData ) {
        let computeCommandBuffer = commandQueue.makeCommandBuffer()
        let computeCommandEncoder = computeCommandBuffer?.makeComputeCommandEncoder()
        
        computeCommandEncoder?.setComputePipelineState(pipelineStateGetDirs)

        computeCommandEncoder?.setBuffer(smplMesh.dirsBuffer, offset: 0, index: 0)
        computeCommandEncoder?.setBuffer(smplMesh.faceIndices, offset: 0, index: 1)
        computeCommandEncoder?.setBuffer(clothMesh.offsetMTLBuffer, offset: 0, index: 2)
        computeCommandEncoder?.setBuffer(smplMesh.vertexBuffer, offset: 0, index: 3)

        let count = smplMesh.faceCount
        let threadExecutionWidth = pipelineStateGetDirs.threadExecutionWidth
        let threadsPerGroup = MTLSize(width: threadExecutionWidth, height: 1, depth: 1)
        let ntg = Int(ceil(Float(count)/Float(threadExecutionWidth)))
        let numThreadgroups = MTLSize(width: ntg, height: 1, depth: 1)
        
        computeCommandEncoder?.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder?.endEncoding()
        computeCommandBuffer?.commit()
    }
    
    func deformSMPL(_ mesh: SMPLMetalMeshData, vertices: MLMultiArray) {
      
        let vector_float3Pointer = vertices.dataPointer.bindMemory(to: vector_float3.self, capacity: Int(vertices.shape[0].int32Value))
//        let pointsList: [vector_float3] = [vector_float3](UnsafeBufferPointer(start: vector_float3Pointer, count: Int(vertices.shape[0].int32Value)))
        vertexBuffer1?.contents().copyMemory(from: vector_float3Pointer, byteCount: vertexBuffer1!.length)

        let computeCommandBuffer = commandQueue.makeCommandBuffer()
        
        
        let computeCommandEncoder = computeCommandBuffer?.makeBlitCommandEncoder()
        computeCommandEncoder?.copy(from: vertexBuffer1!, sourceOffset: 0, to: mesh.vertexBuffer, destinationOffset: 0, size: vertexBuffer1!.length)
        computeCommandEncoder?.endEncoding()
        
        
        let computeCommandEncoder2 = computeCommandBuffer?.makeComputeCommandEncoder()
        
        // STEP 1: INIT NORMALS
        computeCommandEncoder2?.setComputePipelineState(pipelineStateInitNormals)
        computeCommandEncoder2?.setBuffer(mesh.normalBuffer, offset: 0, index: 0)
        var count = mesh.vertexCount
        var threadExecutionWidth = pipelineStateInitNormals.threadExecutionWidth
        var threadsPerGroup = MTLSize(width: threadExecutionWidth, height: 1, depth: 1)
        var ntg = Int(ceil(Float(count)/Float(threadExecutionWidth)))
        var numThreadgroups = MTLSize(width: ntg, height: 1, depth: 1)
        computeCommandEncoder2?.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        // STEP 2: CALCULATE NORMALS PER FACE
        computeCommandEncoder2?.setComputePipelineState(pipelineStateCalcNormals)
        computeCommandEncoder2?.setBuffer(mesh.normalBuffer, offset: 0, index: 0)
        computeCommandEncoder2?.setBuffer(mesh.faceIndices, offset: 0, index: 1)
        computeCommandEncoder2?.setBuffer(mesh.dirsBuffer, offset: 0, index: 2)
        computeCommandEncoder2?.setBuffer(mesh.vertexBuffer, offset: 0, index: 3)
        count = mesh.faceCount
        threadExecutionWidth = pipelineStateCalcNormals.threadExecutionWidth
        threadsPerGroup = MTLSize(width: threadExecutionWidth, height: 1, depth: 1)
        ntg = Int(ceil(Float(count)/Float(threadExecutionWidth)))
        numThreadgroups = MTLSize(width: ntg, height: 1, depth: 1)
        computeCommandEncoder2?.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        // STEP 3: NORMALIZE THE NORMALS PER VERTEX
        computeCommandEncoder2?.setComputePipelineState(pipelineStateNormNormals)
        computeCommandEncoder2?.setBuffer(mesh.normalBuffer, offset: 0, index: 0)
        count = mesh.vertexCount
        threadExecutionWidth = pipelineStateNormNormals.threadExecutionWidth
        threadsPerGroup = MTLSize(width: threadExecutionWidth, height: 1, depth: 1)
        ntg = Int(ceil(Float(count)/Float(threadExecutionWidth)))
        numThreadgroups = MTLSize(width: ntg, height: 1, depth: 1)
        computeCommandEncoder2?.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        
        computeCommandEncoder2?.endEncoding()
        computeCommandBuffer?.commit()
    }
    
    func deformMGM(_ clothMesh: ClothMetalMeshData, smplMesh: SMPLMetalMeshData, vertices: MLMultiArray) {
        // just choose vertInds from vertices
        
        let vector_float3Pointer = vertices.dataPointer.bindMemory(to: vector_float3.self, capacity: Int(vertices.shape[0].int32Value))
        vertexBuffer1?.contents().copyMemory(from: vector_float3Pointer, byteCount: vertexBuffer1!.length)
        
        let computeCommandBuffer = commandQueue.makeCommandBuffer()
        let computeCommandEncoder = computeCommandBuffer?.makeComputeCommandEncoder()
        
        
        computeCommandEncoder?.setComputePipelineState(pipelineStateSlice)

        computeCommandEncoder?.setBuffer(vertexBuffer1!, offset: 0, index: 0)
        computeCommandEncoder?.setBuffer(clothMesh.vertexBuffer, offset: 0, index: 1)
        computeCommandEncoder?.setBuffer(clothMesh.vertInds, offset: 0, index: 2)
        computeCommandEncoder?.setBuffer(smplMesh.normalBuffer, offset: 0, index: 3)
        computeCommandEncoder?.setBuffer(clothMesh.offsetMTLBuffer, offset: 0, index: 4)
        computeCommandEncoder?.setBuffer(smplMesh.vertexBuffer, offset: 0, index: 5)

        let count = clothMesh.vertexCount
        let threadExecutionWidth = pipelineStateSlice.threadExecutionWidth
        let threadsPerGroup = MTLSize(width:threadExecutionWidth, height: 1, depth: 1)
        let ntg = Int(ceil(Float(count)/Float(threadExecutionWidth)))
        let numThreadgroups = MTLSize(width: ntg, height: 1, depth: 1)
        
        computeCommandEncoder?.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeCommandEncoder?.endEncoding()
        computeCommandBuffer?.commit()
        
    }
}

func getDataArray<T>(_ url: URL, def: T) -> Array<T> {
    let data = try? Data(contentsOf: url)
    var ret = Array<T>(repeating: def, count: Int(data!.count/MemoryLayout<T>.size))
    _ = ret.withUnsafeMutableBytes { w in
        data?.copyBytes(to: w)
    }
    return ret
}

/*
Builds a SceneKit geometry object backed by a Metal buffer
*/
class MetalMeshDeformable {

    class func buildSMPL(_ device: MTLDevice, vertices: MLMultiArray, faces: Array<Int32>) -> SMPLMetalMeshData {
        
        let vector_float3Pointer = vertices.dataPointer.bindMemory(to: vector_float3.self, capacity: Int(vertices.shape[0].int32Value))
        let pointsList: [vector_float3] = [vector_float3](UnsafeBufferPointer(start: vector_float3Pointer, count: Int(vertices.shape[0].int32Value)))
        
//        let Int32Pointer = faces.dataPointer.bindMemory(to: Int32.self, capacity: faces.count)
        let indexList: [Int32] = faces // [Int32](UnsafeBufferPointer(start: Int32Pointer, count: faces.count))
        
        let vertexFormat = MTLVertexFormat.float3
        //metal compute shaders cant read and write to same buffer, so make two of them
        //second one could be empty in this case
        let vertexBuffer1 = device.makeBuffer(
            bytes: pointsList,
            length: pointsList.count * MemoryLayout<vector_float3>.size,
            options: [.cpuCacheModeWriteCombined]
        )
        
        let normalBuffer = device.makeBuffer(
            bytes: pointsList,
            length: pointsList.count * MemoryLayout<vector_float3>.size,
            options: [.cpuCacheModeWriteCombined]
        )
        
        let vertexSource = SCNGeometrySource(
            buffer: vertexBuffer1!,
            vertexFormat: vertexFormat,
            semantic: SCNGeometrySource.Semantic.vertex,
            vertexCount: pointsList.count,
            dataOffset: 0,
            dataStride: MemoryLayout<vector_float3>.size)

        let dirs = Array<Float32>(repeating: -1.0, count: indexList.count/3)
        let dirsBuffer = device.makeBuffer(
            bytes: dirs,
            length: dirs.count*MemoryLayout<Float32>.size,
            options: [.cpuCacheModeWriteCombined]
        )
        
//        let colorSource = SCNGeometrySource(
//            buffer: normalBuffer!,
//            vertexFormat: vertexFormat,
//            semantic: .color,
//            vertexCount: pointsList.count,
//            dataOffset: 0,
//            dataStride: MemoryLayout<vector_float3>.size
//        )
        
        let facesBuffer = device.makeBuffer(
            bytes: indexList,
            length: indexList.count*MemoryLayout<Int32>.size,
            options: [.cpuCacheModeWriteCombined]
        )
        let indexData  = Data(bytes: indexList, count: MemoryLayout<Int32>.size * indexList.count)
        let indexElement = SCNGeometryElement(
            data: indexData,
            primitiveType: SCNGeometryPrimitiveType.triangles,
            primitiveCount: indexList.count/3,
            bytesPerIndex: MemoryLayout<Int32>.size
        )
        
        let geo = SCNGeometry(sources: [vertexSource], elements: [indexElement]) // normalSource, uvSource]
        geo.firstMaterial?.isLitPerPixel = true
        
        return SMPLMetalMeshData(
            geometry: geo,
            vertexCount: pointsList.count,
            vertexBuffer: vertexBuffer1!,
            normalBuffer: normalBuffer!,
            faceIndices: facesBuffer!,
            faceCount: Int(indexList.count/3),
            dirsBuffer: dirsBuffer!
        )
    }
    
    class func buildMGM(_ device: MTLDevice, folderName: String, smplVertices: MLMultiArray) -> ClothMetalMeshData {
//        let objName = "garments/\(folderName)/TShirtNoCoat"
//        let texName = "garments/\(folderName)/multi_tex.jpg"
//        let objURL = Bundle.main.url(forResource: "garment_unposed_low_res", withExtension: "obj")
        let folderDir = "Garments/\(folderName)"
        let vertIndsURL = Bundle.main.url(forResource: "vert_inds_low_res", withExtension: "bin", subdirectory: folderDir) // Int32 [N]
        let vertsURL = Bundle.main.url(forResource: "verts_low_res", withExtension: "bin", subdirectory: folderDir) // Int32 [N]
        let displacementsURL = Bundle.main.url(forResource: "displacements_low_res", withExtension: "bin", subdirectory: folderDir) // Float32 [N, 3]
        let objVURL = Bundle.main.url(forResource: "garment_unposed_low_res", withExtension: "vbin", subdirectory: folderDir) // float32 [N, 3]
        let objFURL = Bundle.main.url(forResource: "garment_unposed_low_res", withExtension: "fbin", subdirectory: folderDir) // int32 [N, 3]
        let objVTURL = Bundle.main.url(forResource: "garment_unposed_low_res", withExtension: "vtbin", subdirectory: folderDir)
        let textureURL = Bundle.main.url(forResource: "multi_tex", withExtension: "jpg", subdirectory: folderDir)
        
        let vertIndsData = try? Data(contentsOf: vertIndsURL!) // N*sizeof(Int32) bytes
        let vertsData = try? Data(contentsOf: vertsURL!) // N*sizeof(Int32) bytes
        let displacementsData = try? Data(contentsOf: displacementsURL!) // N*3*sizeof(Float32) bytes
        let vertexData = try? Data(contentsOf: objVURL!)
        let facesData = try? Data(contentsOf: objFURL!)
        let texData = try? Data(contentsOf: objVTURL!)
        
        var vertInds: Array<Int32> = Array<Int32>(repeating: 0, count: Int(vertIndsData!.count/MemoryLayout<Int32>.size))
        var verts = Array<Int32>(repeating: 0, count: Int(vertsData!.count/MemoryLayout<Int32>.size))
        var displacements = Array<Float32>(repeating: 0.0, count: Int(displacementsData!.count/MemoryLayout<Float32>.size))
        var vertices = Array<Float32>(repeating: 0.0, count: Int(vertexData!.count/MemoryLayout<Float32>.size))
        var faces = Array<Int32>(repeating: 0, count: Int(facesData!.count/MemoryLayout<Int32>.size))
        var uvMap = Array<vector_float2>(repeating: vector_float2(repeating: 0.0), count: Int(texData!.count/MemoryLayout<vector_float2>.size))
        print(MemoryLayout<vector_float2>.size, MemoryLayout<vector_float2>.size)
        
        vertInds.withUnsafeMutableBytes { w in
            vertIndsData?.copyBytes(to: w)
        }
        verts.withUnsafeMutableBytes { w in
            vertsData?.copyBytes(to: w)
        }
        displacements.withUnsafeMutableBytes { w in
            displacementsData?.copyBytes(to: w)
        }
        vertices.withUnsafeMutableBytes { w in
            vertexData?.copyBytes(to: w)
        }
        faces.withUnsafeMutableBytes { w in
            facesData?.copyBytes(to: w)
        }
        uvMap.withUnsafeMutableBytes { w in
            texData?.copyBytes(to: w)
        }
        
        // offsets array will be a constant
        // offsets[vertInds] = displacements + noPoseSMPL[verts] - noPoseSMPL[vertInds]
        // offsets must be an MLMultiArray
        var offsets = (try? MLMultiArray(shape: [smplVertices.shape[0], 3], dataType: .float32))!
        let offsetsPointer = UnsafeMutablePointer<Float32>(OpaquePointer(offsets.dataPointer))
        let smplVertsPointer = UnsafeMutablePointer<Float32>(OpaquePointer(smplVertices.dataPointer))
        var absOffsets = Array<vector_float3>(repeating: vector_float3(repeating: 0.0), count: 6890)
        var lhs = 0
        var rhs1 = 0
        var rhs2 = 0
        var lhsOffset = 0
        for i in 0..<vertInds.count {
            lhs = Int(vertInds[i]*offsets.strides[0].int32Value) // + 0*offsets?.strides[1]
            rhs1 = Int(verts[i]*smplVertices.strides[0].int32Value) // + 0*smplVertices.strides[1]
            lhsOffset = Int(verts[i]*offsets.strides[0].int32Value) // + 0*smplVertices.strides[1]
            rhs2 = Int(vertInds[i]*smplVertices.strides[0].int32Value) // + 0*smplVertices.strides[1]
            offsetsPointer[lhsOffset] = displacements[3*i+0] // - smplVertsPointer[rhs1] + smplVertsPointer[rhs2]
            absOffsets[Int(verts[i])].x = offsetsPointer[lhsOffset]

            lhs += offsets.strides[1].intValue
            lhsOffset += offsets.strides[1].intValue
            rhs1 += smplVertices.strides[1].intValue
            rhs2 += smplVertices.strides[1].intValue
            offsetsPointer[lhsOffset] = (displacements[3*i+1]) // - smplVertsPointer[rhs1] + smplVertsPointer[rhs2]
            absOffsets[Int(verts[i])].y = offsetsPointer[lhsOffset]
            
            lhs += offsets.strides[1].intValue
            lhsOffset += offsets.strides[1].intValue
            rhs1 += smplVertices.strides[1].intValue
            rhs2 += smplVertices.strides[1].intValue
            offsetsPointer[lhsOffset] = (displacements[3*i+2]) // - smplVertsPointer[rhs1] + smplVertsPointer[rhs2]
            absOffsets[Int(verts[i])].z = offsetsPointer[lhsOffset]
        }
        
        let offsetMTLBuffer = device.makeBuffer(
            bytes: absOffsets,
            length: absOffsets.count*MemoryLayout<vector_float3>.size,
            options: [.cpuCacheModeWriteCombined]
        )
        
        let vertIndsBuffer = device.makeBuffer(
            bytes: verts, // vertInds,
            length: vertInds.count*MemoryLayout<Int32>.size,
            options: [.cpuCacheModeWriteCombined]
        )
        
        var array3d = Array<vector_float3>(repeating: vector_float3(0.0, 0.0, 0.0), count: Int(vertices.count/3))
        for i in 0..<array3d.count {
            array3d[i] = vector_float3(vertices[3*i], vertices[3*i+1], vertices[3*i+2])
        }
        let vertexBuffer1 = device.makeBuffer(
            bytes: array3d,
            length: array3d.count*MemoryLayout<vector_float3>.size,
            options: [.cpuCacheModeWriteCombined]
        )
        print(array3d.count)
        let vertexSource = SCNGeometrySource(
            buffer: vertexBuffer1!,
            vertexFormat: MTLVertexFormat.float3,
            semantic: SCNGeometrySource.Semantic.vertex,
            vertexCount: array3d.count,
            dataOffset: 0,
            dataStride: MemoryLayout<vector_float3>.size
        )
        
        let texBuffer = device.makeBuffer(
            bytes: uvMap,
            length: uvMap.count*MemoryLayout<vector_float2>.size,
            options: [.cpuCacheModeWriteCombined]
        )
        let uvSource = SCNGeometrySource(
            buffer: texBuffer!,
            vertexFormat: MTLVertexFormat.float2,
            semantic: SCNGeometrySource.Semantic.texcoord,
            vertexCount: uvMap.count,
            dataOffset: 0,
            dataStride: MemoryLayout<vector_float2>.size
        )
        
        // let uvSource = tshirtNoCoat_geo.sources(for: .texcoord)[0]
        
        let indexData  = Data(bytes: faces, count: MemoryLayout<Int32>.size * faces.count)
        let indexElement = SCNGeometryElement(
            data: indexData,
            primitiveType: SCNGeometryPrimitiveType.triangles,
            primitiveCount: faces.count/3,
            bytesPerIndex: MemoryLayout<Int32>.size
        )
        
        let geo = SCNGeometry(sources: [vertexSource, uvSource], elements: [indexElement])
        geo.firstMaterial?.diffuse.contents = textureURL! // UIImage(contentsOfFile: textureURL!.absoluteString)

        return ClothMetalMeshData(
            geometry: geo,
            vertexCount: array3d.count,
            vertexBuffer: vertexBuffer1!,
            offsetMLArray: offsets,
            vertInds: vertIndsBuffer!,
            offsets: offsetMTLBuffer!
        )
    }

}
