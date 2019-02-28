using LinearAlgebra
using BenchmarkTools
using StaticArrays

function position_in_world(x::Float64, y::Float64)
      target_width_px = 1123
      target_height_px = 791
      target_width_mm = 297
      target_height_mm = 210.025
      X = ((x + 0.5) / target_width_px) * target_width_mm - 0.5 * target_width_mm
      Y = ((y + 0.5) / target_height_px) * target_height_mm - 0.5 * target_height_mm
      Z = 0.0
      return @SVector [X, Y, Z ]
end

function π_projection_function(K, R, t, point)
      # point_in_camera = R * point + t
      point_in_image = K * (R * point + t)
      return point_in_image / point_in_image[end]
end

function compute_homograpy_DLT(M, P)
      # A = zeros(Float64, 8, 9)
      A = @SMatrix zeros(8,9)
      # print(size(M)[1])
      for i in 1:size(M)[1]
            A[2 * i - 1, 1:3] = @view M[i,:]
            A[2 * i - 1, 7:9] = - P[i,1] * @view M[i,:]
            A[2 * i, 4:6] = @view M[i,:]
            A[2 * i, 7:9] = - P[i,2] * @view M[i,:]
      end
      # print(A)
      return nullspace(A)
      # return collect(1:9)
end

function main()
      K = @SMatrix [ 1169.19630        0.0 652.98743;
            0.0 1169.61014 528.83429;
            0.0        0.0       1.0]

      T = @SMatrix [0.961255 -0.275448 0.0108487    112.79;
       0.171961  0.629936   0.75737  -217.627;
       -0.21545  -0.72616  0.652895   1385.13]

      R = @view T[1:3, 1:3]
      t = @view T[:,end]

      tl = position_in_world(0.0, 0.0)
      tr = position_in_world(1123.0, 0.0)
      br = position_in_world(1123.0, 791.0)
      bl = position_in_world(0.0, 791.0)

      p1 = π_projection_function(K, R, t, tl)
      p2 = π_projection_function(K, R, t, tr)
      p3 = π_projection_function(K, R, t, br)
      p4 = π_projection_function(K, R, t, bl)
      P = [p1 p2 p3 p4]'

      M = @SMatrix [0.0     0.0    1.0;
                    1123.0  0.0    1.0;
                    1123.0  791.0  1.0;
                    0.0     791.0  1.0]

      res = compute_homograpy_DLT(M , P)
      homography = reshape(res, (3,3))'
      println("Homography: ", homography)
end

@btime main()
