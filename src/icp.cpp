#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "icp.hpp"

// Euclidean distance
float ICP::dist(const Vector3d &a, const Vector3d &b)
{
	return sqrt((a[0] - b[0])*(a[0] - b[0]) + (a[1] - b[1])*(a[1] - b[1]) + (a[2] - b[2])*(a[2] - b[2]));
}

void ICP::set_maximum_iterations(int iter)
{
    max_iter = iter;
}

Matrix4d ICP::best_fit_transform_SVD(const MatrixXd &A, const MatrixXd &B)
{
	size_t num_rows = std::min(A.rows(), B.rows());

	Vector3d centroid_A(0, 0, 0);
	Vector3d centroid_B(0, 0, 0);
	
	MatrixXd AA;
	MatrixXd BB;
	if(A.rows() > B.rows())
    {
        AA = BB = B;
    }
	else
    {
        BB = AA = A;
    }

	Matrix4d T = MatrixXd::Identity(4, 4);

	for(int i = 0; i < num_rows; ++i)
	{
		centroid_A += A.block<1,3>(i,0).transpose();
		centroid_B += B.block<1,3>(i,0).transpose();
	}

	centroid_A /= num_rows;
	centroid_B /= num_rows;

	for(int i = 0; i < num_rows; ++i)
	{
		AA.block<1,3>(i, 0) = A.block<1,3>(i, 0) - centroid_A.transpose();
		BB.block<1,3>(i, 0) = B.block<1,3>(i, 0) - centroid_B.transpose();
	}

	MatrixXd H = AA.transpose()*BB;
	MatrixXd U;
	VectorXd S;
	MatrixXd V;
	MatrixXd Vt;
	Matrix3d R;
	Vector3d t;
	
	JacobiSVD<MatrixXd> svd(H, ComputeFullU | ComputeFullV);
	// JacobiSVD decomposition computes only the singular values by default
	// ComputeFullU or ComputeFullV : ask for U or V explicitly
	U = svd.matrixU();
	S = svd.singularValues();
	V = svd.matrixV();
	Vt = V.transpose();

    R = Vt.transpose()*U.transpose();

	if(R.determinant() < 0)
	{
		Vt.block<1,3>(2, 0) *= -1;
        R = Vt.transpose()*U.transpose();
	}

	t = centroid_B - R * centroid_A;

	T.block<3,3>(0, 0) = R;
	T.block<3,1>(0, 3) = t;

	// std::cout << "\n" << T << "\n" << std::endl;

	return T;
}

Matrix4d ICP::best_fit_transform_quat(const MatrixXd &A, const MatrixXd &B)
{
	size_t num_rows = std::min(A.rows(), B.rows());

	Vector3d centroid_A(0, 0, 0);
	Vector3d centroid_B(0, 0, 0);
	
	MatrixXd AA;
	MatrixXd BB;
	if(A.rows() > B.rows())
    {
        AA = BB = B;
    }
	else
    {
        BB = AA = A;
    }

	Matrix4d T = MatrixXd::Identity(4, 4);

	for(int i = 0; i < num_rows; ++i)
	{
		centroid_A += A.block<1,3>(i, 0).transpose();
		centroid_B += B.block<1,3>(i, 0).transpose();
	}

	centroid_A /= num_rows;
	centroid_B /= num_rows;

	for(int i = 0; i < num_rows; ++i)
	{
		AA.block<1,3>(i, 0) = A.block<1,3>(i, 0) - centroid_A.transpose();
		BB.block<1,3>(i, 0) = B.block<1,3>(i, 0) - centroid_B.transpose();
	}

	MatrixXd H = AA.transpose()*BB; // 3x3 covariance matrix
	MatrixXd As = H - H.transpose(); // anti-symmetric matrix

	Vector3d delta(As(1, 2), As(2, 0), As(0, 1));

	MatrixXd Q = MatrixXd::Zero(4, 4);
	MatrixXd temp = MatrixXd::Zero(3, 3);
	MatrixXd traceEye = temp;
	double cov_trace = H.trace();

	for(int r = 0; r < traceEye.rows(); ++r) 
	{
		for(int c = 0; c < traceEye.cols(); ++c) 
		{
			if(r == c) 
			{
				traceEye(r, c) = cov_trace;
			}
		}
	}

	temp = H + H.transpose() - traceEye;

	Q(0, 0) = cov_trace;
	Q.block<1,3>(0, 1) = delta.transpose();
	Q.block<3,1>(1, 0) = delta;	

	Q.block<1,3>(1, 1) = temp.block<1,3>(0, 0);
	Q.block<1,3>(2, 1) = temp.block<1,3>(1, 0);
	Q.block<1,3>(3, 1) = temp.block<1,3>(2, 0);

	EigenSolver<MatrixXd> es(Q);
	VectorXd e_vals = es.eigenvalues().real();
	MatrixXd e_vecs = es.eigenvectors().real();

	// std::cout << "The eigenvalues of Q are:" << std::endl << e_vals << std::endl;
	// std::cout << "The matrix of eigenvectors, V, is:" << std::endl << e_vecs << std::endl << std::endl;

	// get location of maximum
  	Eigen::Index max_row, max_col;
  	// std::complex<double> max = e_vals.maxCoeff(&maxRow, &maxCol);
	double max = e_vals.maxCoeff(&max_row, &max_col);

	// std::cout << "The max eigenvalue: " << max << ", with position: " << maxRow << "," << maxCol << std::endl;

	VectorXd max_e_vecs = e_vecs.col(max_row);

	MatrixXd R = MatrixXd::Zero(3, 3);

	double q0 = max_e_vecs(0, 0);
	double q1 = max_e_vecs(1, 0);
	double q2 = max_e_vecs(2, 0);
	double q3 = max_e_vecs(3, 0);

	R(0, 0) = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3;
	R(0, 1) = 2 * (q1 * q2 - q0 * q3);
	R(0, 2) = 2 * (q1 * q3 + q0 * q2);

	R(1, 0) = 2 * (q1 * q2 + q0 * q3);
	R(1, 1) = q0 * q0 + q2 * q2 - q1 * q1 - q3 * q3;
	R(1, 2) = 2 * (q2 * q3 - q0 * q1);

	R(2, 0) = 2 * (q1 * q3 - q0 * q2);
	R(2, 1) = 2 * (q2 * q3 + q0 * q1);
	R(2, 2) = q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2;

	Vector3d t = Vector3d::Zero(3, 1);

	t = centroid_B - (R * centroid_A);

	T.block<3,3>(0, 0) = R;
	T.block<3,1>(0, 3) = t;

	std::cout << "\n" << T << "\n" << std::endl;

	return T;
}

ICP_OUT ICP::icp_alg(const MatrixXd &A, const MatrixXd &B, int max_iteration, float tolerance, int leaf_size, int Ksearch)
{
	auto start = std::chrono::high_resolution_clock::now();

	size_t row = std::min(A.rows(), B.rows());
	MatrixXd src = MatrixXd::Ones(3+1, row);
	MatrixXd src3d = MatrixXd::Ones(3, row);
	MatrixXd dst = MatrixXd::Ones(3+1, B.rows());
    NEIGHBORS neighbor;
  	Matrix4d T;
    Eigen::MatrixXd dst_chorder = Eigen::MatrixXd::Ones(3, row);
	ICP_OUT result;
	int iter;

	for(int i = 0; i < row; i++)
	{
		src.block<3,1>(0, i) = A.block<1,3>(i, 0).transpose();
		src3d.block<3,1>(0, i) = A.block<1,3>(i, 0).transpose(); // save the temp data
	}

	for(int i = 0; i < B.rows(); ++i)
	{
		dst.block<3,1>(0, i) = B.block<1,3>(i, 0).transpose();
	}

	double prev_error = 0;
	double mean_error = 0;

	// when the number of iterations is less than the maximum
	for(iter = 1; iter <= max_iteration; ++iter)
	{
		std::cout << "----------------" << std::endl;
		std::cout << "Iteration: " << iter << std::endl;

        // neighbor = nearest_neighbor_naive(src3d.transpose(), B);
		neighbor = nearest_neighbor_kdtree(src3d.transpose(), B);

        for(int j = 0; j < row; ++j)
		{
            dst_chorder.block<3,1>(0, j) = dst.block<3,1>(0, neighbor.B_indices[j]);
        }

		// save the transformed matrix in this iteration
		T = best_fit_transform_quat(src3d.transpose(), dst_chorder.transpose());
        // T = best_fit_transform_SVD(src3d.transpose(), dst_chorder.transpose());
		src = T*src;

		// copy the transformed matrix
		for(int j = 0; j < row; ++j)
        {
            src3d.block<3,1>(0, j) = src.block<3,1>(0, j);
        }

        mean_error = sqrt(std::accumulate(neighbor.distances.begin(), neighbor.distances.end(), 0.0)/neighbor.distances.size());

        std::cout << "Mean distance error: " << abs(prev_error - mean_error) <<std::endl;

        if(abs(prev_error - mean_error) < tolerance)
        {
			std::cout << "ICP has converged in: " << iter << " iterations" <<std::endl;
            break;
        }
        prev_error = mean_error;
	}

	if(iter == max_iter+1)
	{
		std::cout << "ICP has reached maximum iterations" <<std::endl;
	}

	auto finish = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
	std::cout << "Convergence in " << static_cast<double>(duration.count())/1000 << " milliseconds and " << iter << " iterations" << std::endl;

    T = best_fit_transform_quat(A, src3d.transpose());

	result.trans_mat = T;
	result.distances = neighbor.distances;
	result.iter = iter;

	return result;
}

ICP_OUT ICP::tr_icp_alg(const MatrixXd &A, const MatrixXd &B, int max_iteration, float tolerance, float min_mse, int leaf_size, int Ksearch)
{
	auto start = std::chrono::high_resolution_clock::now();

	size_t row = std::min(A.rows(), B.rows());
	MatrixXd src = MatrixXd::Ones(3+1, row);
	MatrixXd src3d = MatrixXd::Ones(3, row);
	MatrixXd dst = MatrixXd::Ones(3+1, B.rows());
    NEIGHBORS neighbor;
  	Matrix4d T;
    Eigen::MatrixXd dst_trim;
	Eigen::MatrixXd src_trim;
	ICP_OUT result;
	int iter;

	for(int i = 0; i < row; ++i)
	{
		src.block<3,1>(0,i) = A.block<1,3>(i,0).transpose();
		src3d.block<3,1>(0,i) = A.block<1,3>(i,0).transpose(); // save the temp data
	}

	for(int i = 0; i < B.rows(); ++i)
	{
		dst.block<3,1>(0, i) = B.block<1,3>(i, 0).transpose();
	}

	double prev_error = 0;
	double mse = 0;

	// when the number of iterations is less than the maximum
	for(iter = 1; iter <= max_iteration; ++iter)
	{
		std::cout << "----------------" << std::endl;
		std::cout << "Iteration: " << iter << std::endl;

        // neighbor = nearest_neighbor_naive(src3d.transpose(), B);
		neighbor = nearest_neighbor_kdtree(src3d.transpose(), B);

		// sort vectors
		std::vector<float> dist = neighbor.distances;
		std::vector<int> dst_ind = neighbor.B_indices; // closest indicies from model dataset
		std::vector<int> src_ind = neighbor.A_indices;

		auto p = sort_permutation(dist, [](float const& a, float const& b){ return a < b; });

		neighbor.distances = apply_permutation(dist, p);
		neighbor.B_indices = apply_permutation(dst_ind, p);
		neighbor.A_indices = apply_permutation(src_ind, p);

		double o = get_overlap_parameter(neighbor.distances);
		int trimmed_length = (int)(o*row);

		std::cout << "Number of points: " << row << std::endl;
		std::cout << "Number of points (trimmed): " << trimmed_length << std::endl;

		dst_trim = Eigen::MatrixXd::Ones(3, trimmed_length);
		src_trim = Eigen::MatrixXd::Ones(3, trimmed_length);

        for(int j = 0; j < trimmed_length; ++j)
		{
            dst_trim.block<3,1>(0, j) = dst.block<3,1>(0, neighbor.B_indices[j]);
			src_trim.block<3,1>(0, j) = src.block<3,1>(0, neighbor.A_indices[j]);
        }

		// save the transformed matrix in this iteration
		T = best_fit_transform_quat(src_trim.transpose(), dst_trim.transpose());
        // T = best_fit_transform_SVD(src3d.transpose(), dst_chorder.transpose());
		src = T*src;

		// copy the transformed matrix
		for(int j = 0; j < row; ++j)
        {
            src3d.block<3,1>(0, j) = src.block<3,1>(0, j);
        }

        double mean_error = sqrt(std::accumulate(neighbor.distances.begin(), neighbor.distances.end(), 0.0)/neighbor.distances.size());
		mse = sqrt(trimmed_mse(o, neighbor.distances));

        std::cout << "Mean distance error: " << mean_error <<std::endl;
		std::cout << "Trimmed MSE: " << mse <<std::endl;

        if(abs(prev_error - mse) < tolerance)
        {
			std::cout << "TrICP has converged in: " << iter << " iterations due to small relative change in error" <<std::endl;
            break;
        }
		if(mse < min_mse)
        {
			std::cout << "TrICP has converged in: " << iter << " iterations" <<std::endl;
            break;
        }
        prev_error = mse;
	}

	if(iter == max_iter+1)
	{
		std::cout << "TrICP has reached maximum iterations" <<std::endl;
	}

	auto finish = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(finish - start);
	std::cout << "Convergence in " << static_cast<double>(duration.count())/1000 << " milliseconds and " << iter << " iterations" << std::endl;

    T = best_fit_transform_quat(A, src3d.transpose());

	result.trans_mat = T;
	result.distances = neighbor.distances;
	result.iter = iter;

	return result;
}

void ICP::align(pcl::PointCloud<pcl::PointXYZ>& cloud_icp_)
{
    MatrixXf source_matrix = cloud_icp->getMatrixXfMap(3, 4, 0).transpose();
    MatrixXf target_matrix = cloud_in->getMatrixXfMap(3, 4, 0).transpose();

    float tolerance = 0.000001;
	float min_mse = 0.000001;

    // call icp
    // ICP_OUT icp_result = icp_alg(source_matrix.cast<double>(), target_matrix.cast<double>(), max_iter, tolerance);
	ICP_OUT icp_result = tr_icp_alg(source_matrix.cast<double>(), target_matrix.cast<double>(), max_iter, tolerance, min_mse);

    Matrix4f T = icp_result.trans_mat.cast<float>();

    MatrixXf source_trans_matrix = source_matrix;
    
    int num_rows = std::min(source_matrix.rows(), target_matrix.rows());
    MatrixXf source_trans4d = MatrixXf::Ones(3+1, num_rows);
    for(int i = 0; i < num_rows; ++i)
    {
        source_trans4d.block<3,1>(0, i) = source_matrix.block<1,3>(i, 0).transpose();
    }
    source_trans4d = T*source_trans4d;
    for(int i = 0; i < num_rows; ++i)
    {
        source_trans_matrix.block<1,3>(i, 0) = source_trans4d.block<3,1>(0, i).transpose();
    }

    pcl::PointCloud<pcl::PointXYZ> temp_cloud;
    temp_cloud.width = num_rows;
    temp_cloud.height = 1;
    temp_cloud.points.resize(num_rows);
    for(size_t n = 0; n < num_rows; ++n)
    {
        temp_cloud[n].x = source_trans_matrix(n, 0);
        temp_cloud[n].y = source_trans_matrix(n, 1);
        temp_cloud[n].z = source_trans_matrix(n, 2);
    }
    cloud_icp_ = temp_cloud;
}

NEIGHBORS ICP::nearest_neighbor_naive(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
{
    int num_rows_A = A.rows();
    int num_rows_B = B.rows();
    Eigen::Vector3d vec_A;
    Eigen::Vector3d vec_B;
    NEIGHBORS neigh;
    float min = 100;
    int index = 0;
    float dist_temp = 0;

    for(int i = 0; i < num_rows_A; ++i)
    {
        vec_A = A.block<1,3>(i, 0).transpose();
        min = 100;
        index = 0;
        dist_temp = 0;

        for(int j = 0; j < num_rows_B; ++j)
        {
            vec_B = B.block<1,3>(j, 0).transpose();
            dist_temp = dist(vec_A, vec_B);
            if(dist_temp < min)
            {
                min = dist_temp;
                index = j;
            }
        }

        neigh.distances.push_back(min);
        neigh.B_indices.push_back(index);
		neigh.A_indices.push_back(i);
    }

    return neigh;
}

NEIGHBORS ICP::nearest_neighbor_kdtree(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
{
    int num_rows_A = A.rows();
    NEIGHBORS neigh;

	using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    matrix_t mat = B;

	using my_kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<matrix_t, SAMPLES_DIM /*fixed size*/, nanoflann::metric_L2>;
	my_kd_tree_t mat_index(SAMPLES_DIM, std::cref(mat), 10 /* max leaf */);

	for(int i = 0; i < num_rows_A; ++i)
	{
		std::vector<double> query_pt(SAMPLES_DIM);
		for(size_t d = 0; d < SAMPLES_DIM; ++d)
		{
			query_pt[d] = A(i, d);
		}

		// do a knn search
    	const size_t num_results = 1;
    	std::vector<size_t> ret_indexes(num_results);
    	std::vector<double> out_dists_sqr(num_results);

		nanoflann::KNNResultSet<double> resultSet(num_results);
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

		mat_index.index_->findNeighbors(resultSet, &query_pt[0]);		

		// neigh.distances.push_back(sqrt(out_dists_sqr[0]));
		neigh.distances.push_back(out_dists_sqr[0]);
        neigh.B_indices.push_back(ret_indexes[0]);
		neigh.A_indices.push_back(i);
	}

	return neigh;
}

double ICP::get_overlap_parameter(const std::vector<float> &distances)
{
	double min_overlap = 0.2;
	double lambda = 2.0;
	double obj_fun = 0;
	double obj_fun_next = 0;
	double obj_fun_prev = 0;
	double overlap_step = 0.01;

	while(min_overlap <= 1.0)
	{
		obj_fun = trimmed_mse(min_overlap, distances) * (1/pow(min_overlap, lambda + 1));
		obj_fun_prev = trimmed_mse(min_overlap - overlap_step, distances) * (1/pow((min_overlap - overlap_step), lambda + 1));
		obj_fun_next = trimmed_mse(min_overlap + overlap_step, distances) * (1/pow((min_overlap + overlap_step), lambda + 1));

		if((obj_fun < obj_fun_prev) && (obj_fun_next > obj_fun))
		{
			return min_overlap;
			break;
		}
		else 
		{
			min_overlap += overlap_step;
		}
	}
	return min_overlap;
}

double ICP::trimmed_mse(const double &overlap, const std::vector<float> &distances)
{
	double trimmed_mse = 0;
	int length = overlap * distances.size();
	for(int i = 0; i < length; ++i)
	{
		trimmed_mse += distances[i];
	}
	trimmed_mse /= length;
	return trimmed_mse;
}