#include <iostream>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Dense>

Eigen::Vector3d position_in_world(double x, double y)
{
      const double target_width_px = 1123;
      const double target_height_px = 791;
      const double target_width_mm = 297;
      const double target_height_mm = 210.025;
      const double X = ((x + 0.5) / target_width_px) * target_width_mm - 0.5 * target_width_mm;
      const double Y = ((y + 0.5) / target_height_px) * target_height_mm - 0.5 * target_height_mm;
      const double Z = 0.0;
      return Eigen::Vector3d(X, Y, Z);
}

Eigen::Vector3d phi_projection_function(const Eigen::Matrix3d& K, 
                                        const Eigen::Matrix3d& R, 
                                        const Eigen::Vector3d& t, 
                                        const Eigen::Vector3d& point)
{
    //   const Eigen::Vector3d point_in_camera = R * point + t;
      const Eigen::Vector3d point_in_image = K * (R * point + t);
      return point_in_image / point_in_image(2);
}

Eigen::VectorXd compute_homograpy_DLT(const Eigen::MatrixXd& M, const Eigen::MatrixXd& P)
{
      Eigen::MatrixXd A (8, 9);
      A.setZero();
      for(int32_t i(0); i < M.rows(); i++)
      {
          A.block(2 * i, 0, 1, 3) = M.row(i);
          A.block(2 * i, 6, 1, 3) = - P.row(i)(0) * M.row(i);
          A.block(2 * i + 1, 3, 1, 3) = M.row(i);
          A.block(2 * i + 1, 6, 1, 3) = - P.row(i)(1) * M.row(i);
      }
    //   std::cout << "A: \n" << A << std::endl;
    Eigen::JacobiSVD< Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner > svd_null(
          A, Eigen::ComputeFullV );

    // std::cout << "V: " << svd_null.matrixV() << std::endl;
    return svd_null.matrixV().col(8);
}


int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    const int repeat = 100000;
    for (int i(0); i < repeat; i++)
    {
        Eigen::Matrix3d K;
        K << 1169.19630, 0.0, 652.98743, 0.0, 1169.61014, 528.83429, 0.0, 0.0, 1.0;
        Eigen::MatrixXd T(3,4);
        T << 0.961255, -0.275448, 0.0108487,    112.79, 0.171961,  0.629936,   0.75737,  -217.627,
            -0.21545,  -0.72616,  0.652895,   1385.13;
        // std::cout << "K: " << K << std::endl;
        // std::cout << "T: " << T << std::endl;

        const auto& R = T.block(0,0,3,3);
        const auto& t = T.col(3);
        // auto end1 = std::chrono::high_resolution_clock::now();
        // std::cout << "elapsed time K,T,R,t (ns): "
        //               << std::chrono::duration_cast< std::chrono::nanoseconds >( end1 - start ).count() << std::endl;
        // std::cout << "R: " << R << std::endl;
        // std::cout << "t: " << t << std::endl;

        const Eigen::Vector3d tl = position_in_world(0.0, 0.0);
        const Eigen::Vector3d tr = position_in_world(1123.0, 0.0);
        const Eigen::Vector3d br = position_in_world(1123.0, 791.0);
        const Eigen::Vector3d bl = position_in_world(0.0, 791.0);
        // std::cout << "tl: " << tl.transpose() << std::endl;
        // std::cout << "tr: " << tr.transpose() << std::endl;
        // std::cout << "br: " << br.transpose() << std::endl;
        // std::cout << "bl: " << bl.transpose() << std::endl;
        // auto end2 = std::chrono::high_resolution_clock::now();
        // std::cout << "elapsed time function point in world (ns): "
        //               << std::chrono::duration_cast< std::chrono::nanoseconds >( end2 - start ).count() << std::endl;

        const Eigen::Vector3d p1 = phi_projection_function(K, R, t, tl);
        const Eigen::Vector3d p2 = phi_projection_function(K, R, t, tr);
        const Eigen::Vector3d p3 = phi_projection_function(K, R, t, br);
        const Eigen::Vector3d p4 = phi_projection_function(K, R, t, bl);
        // auto end3 = std::chrono::high_resolution_clock::now();
        // std::cout << "elapsed time function phi (ns): "
        //               << std::chrono::duration_cast< std::chrono::nanoseconds >( end3 - start ).count() << std::endl;

        Eigen::MatrixXd P(4, 3);
        P.row(0) = p1;
        P.row(1) = p2;
        P.row(2) = p3;
        P.row(3) = p4;
        // std::cout << "P: \n" << P << std::endl;

        const Eigen::Vector3d m1(0.0, 0.0, 1.0);
        const Eigen::Vector3d m2(1123.0, 0.0, 1.0);
        const Eigen::Vector3d m3(1123.0, 791.0, 1.0);
        const Eigen::Vector3d m4(0.0, 791.0, 1.0);
        Eigen::MatrixXd M(4,3);
        M.row(0) = m1;
        M.row(1) = m2;
        M.row(2) = m3;
        M.row(3) = m4;
        // std::cout << "M: \n" << M << std::endl;

        Eigen::VectorXd res = compute_homograpy_DLT(M, P);
        // std::cout << "res: " << res.transpose() << std::endl;
        Eigen::Map < Eigen::Matrix3d > homography(res.data(), 3, 3);
        homography.transposeInPlace();
        std::cout << "Homography: \n" << homography << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double tt = std::chrono::duration_cast< std::chrono::microseconds >( end - start ).count();
    std::cout << "elapsed time (ns): "
                  << tt / repeat << std::endl;
    return 0;
}