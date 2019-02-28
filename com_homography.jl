using LinearAlgebra

function position_in_world(x::Float64, y::Float64)
      target_width_px = 1123
      target_height_px = 791
      target_width_mm = 297
      target_height_mm = 210.025
      X = ((x + 0.5) / target_width_px) * target_width_mm - 0.5 * target_width_mm
      Y = ((y + 0.5) / target_height_px) * target_height_mm - 0.5 * target_height_mm
      Z = 0.0
      return [X, Y, Z ]
end

# function position_in_world2!(x::Float64, y::Float64, vec::Array{Float64,1})
#       target_width_px = 1123
#       target_height_px = 791
#       target_width_mm = 297
#       target_height_mm = 210.025
#       vec[1] = ((x + 0.5) / target_width_px) * target_width_mm - 0.5 * target_width_mm
#       vec[2] = ((y + 0.5) / target_height_px) * target_height_mm - 0.5 * target_height_mm
#       vec[3] = 0.0
# end


function π_projection_function(K::Array{Float64,2}, R::Array{Float64,2}, t::Array{Float64,1}, point::Array{Float64,1})
      point_in_camera = R * point + t
      point_in_image = K * point_in_camera
      return point_in_image / point_in_camera[end]
end


function compute_homograpy_DLT(M, P)
      A = zeros(Float64, 8, 9)
      # print(size(M)[1])
      for i in 1:size(M)[1]
            A[2 * i - 1, 1:3] = M[i,:]
            A[2 * i - 1, 7:9] = - P[i,1] * M[i,:]
            A[2 * i, 4:6] = M[i,:]
            A[2 * i, 7:9] = - P[i,2] * M[i,:]
      end
      # print(A)
      return nullspace(A)
      # return collect(1:9)
end

@timev begin
K = [ 1169.19630        0.0 652.98743;
      0.0 1169.61014 528.83429;
      0.0        0.0       1.0]

T = [0.961255 -0.275448 0.0108487    112.79;
 0.171961  0.629936   0.75737  -217.627;
 -0.21545  -0.72616  0.652895   1385.13]

R = T[1:3, 1:3]
t = T[:,end]

tl = position_in_world(0.0, 0.0)
tr = position_in_world(1123.0, 0.0)
br = position_in_world(1123.0, 791.0)
bl = position_in_world(0.0, 791.0)

# println("pos in world 2")
# @timev begin
# tl = Float64[0.0, 0.0, 0.0]
# tr = Float64[0.0, 0.0, 0.0]
# br = Float64[0.0, 0.0, 0.0]
# bl = Float64[0.0, 0.0, 0.0]
# position_in_world2!(0.0, 0.0,tl)
# position_in_world2!(1123.0, 0.0,tr)
# position_in_world2!(1123.0, 791.0,br)
# position_in_world2!(0.0, 791.0,bl)
# end


p1 = π_projection_function(K, R, t, tl)
p2 = π_projection_function(K, R, t, tr)
p3 = π_projection_function(K, R, t, br)
p4 = π_projection_function(K, R, t, bl)
P = [p1 p2 p3 p4]'

m1 = [0.0, 0.0, 1.0]
m2 = [1123.0, 0.0, 1.0]
m3 = [1123.0, 791.0, 1.0]
m4 = [0.0, 791.0, 1.0]
M = [m1 m2 m3 m4]'

res = compute_homograpy_DLT(M , P)
homography = reshape(res, (3,3))'
println("Homography: ", homography)
end
