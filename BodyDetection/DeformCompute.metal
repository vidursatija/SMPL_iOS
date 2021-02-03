//
//  DeformCompute.metal
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

#include <metal_stdlib>
using namespace metal;


// ONLY TO BE CALCULATED ONCE PER FACE
kernel void dirsPerFaceNormals(device float *dirs [[ buffer(0) ]],
                           const device int *faceArray [[ buffer(1) ]],
                           const device float3 *offsetArray [[ buffer(2) ]],
                           const device float3 *bodyVertices [[ buffer(3) ]],
                           uint id [[ thread_position_in_grid ]]
                           ) {
    int i = 3*id;
    const float3 offsetVector = offsetArray[faceArray[i]] + offsetArray[faceArray[i+1]] + offsetArray[faceArray[i+2]];
    const float offsetLength = fast::length(offsetVector);
    if(offsetLength > 1e-6)
        return;
    const float3 v12 = bodyVertices[faceArray[i]] - bodyVertices[faceArray[i+1]];
    const float3 v32 = bodyVertices[faceArray[i+2]] - bodyVertices[faceArray[i+1]];
    float3 norm = cross(v12, v32);
    float direction = dot(norm, offsetVector);
    if(direction > 0)
        dirs[id] = 1.0;
    else
        dirs[id] = -1.0;
}

// NEXT 3 FUNCTIONS CALLED EVERY FRAME
kernel void initializeNormals(device float3 *normVerts [[buffer(0) ]],
                              uint id [[ thread_position_in_grid ]]
                              ) {
    normVerts[id] = float3(0.0);
}

kernel void perFaceNormals(device float3 *normVerts [[ buffer(0) ]],
                           const device int *faceArray [[ buffer(1) ]],
                           const device float *dirs [[ buffer(2) ]],
                           const device float3 *bodyVertices [[ buffer(3) ]],
                           uint id [[ thread_position_in_grid ]]
                           ) {
    int i = 3*id;
    const float3 v12 = bodyVertices[faceArray[i]] - bodyVertices[faceArray[i+1]];
    const float3 v32 = bodyVertices[faceArray[i+2]] - bodyVertices[faceArray[i+1]];
    const float3 norm = fast::normalize(dirs[id]*cross(v12, v32));
    normVerts[faceArray[i]] += norm;
    normVerts[faceArray[i+1]] += norm;
    normVerts[faceArray[i+2]] += norm;
}

kernel void perVextexNormalize(device float3 *normVerts [[ buffer(0) ]],
                               uint id [[ thread_position_in_grid ]]
                               ) {
    normVerts[id] = fast::normalize(normVerts[id]);
}

// TO SLICE CLOTHED SMPL EVERY FRAME
kernel void sliceVertex(const device float3 *inVerts [[ buffer(0) ]], // BODY WITH CLOTH VERTICES 6890
                        device float3 *outVerts [[ buffer(1) ]], // TSHIRT LEN
                        const device int *vertInds [[ buffer(2) ]], // TSHIRT LEN
                        const device float3 *inVertsNormals [[ buffer(3) ]], // BODY VERTEX NORMALS 6890
                        const device float3 *offsetArray [[ buffer(4) ]], // CLOTH - BODY OFFSETS 6890
                        const device float3 *smplVertices [[ buffer(5) ]], // BODY VERTICES 6890
                        uint id [[ thread_position_in_grid ]])
{
    const int idOnBody = vertInds[id];
//    if(dot(inVerts[idOnBody] - smplVertices[idOnBody], inVertsNormals[idOnBody]) > 0)
//        outVerts[id] = 0.1*inVerts[idOnBody] + 0.9*(smplVertices[idOnBody] + 0.015*inVertsNormals[idOnBody]); // inVerts[idOnBody]; // inVerts[idOnBody];
//    else
    outVerts[id] = smplVertices[idOnBody] + inVertsNormals[idOnBody]*length(offsetArray[idOnBody]);
}
