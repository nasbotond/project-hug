#if __INTELLISENSE__
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#include "icp.hpp"

/*
	Compute the distance between two 1*3 vector
	sqrt(sum(a[i]-b[i])^2), i:=0,1,2
*/
float ICP::dist(const Vector3d &a, const Vector3d &b)
{
	return sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]));
}

void ICP::setMaximumIterations(int iter)
{
    max_iter = iter;
}


/*
	Transform A to align best with B
	(B has be correspondented to A)
	Input: 
		source     	A = [x1,y1,z1]
						|x2,y2,z2|
						|........|
						[xn,yn,zn]
		destination B = [x1',y1',z1']
						|x2',y2',z2'|
						|...........|
						[xn',yn',zn']
					  # [xi',yi',zi'] is the nearest to [xi,yi,zi]
		
	Output: 
		T = [R, t]
			 R - rotation: 3*3 matrix
			 t - tranlation: 3*1 vector
	"best align" equals to find the min value of 
					sum((bi-R*ai-t)^2)/N, i:=1~N
	the solution is:
		centroid_A = sum(ai)/N, i:=1~N
		centroid_B = sum(bi)/N, i:=1~N
		AA = {ai-centroid_A}
		BB = {bi-centroid_B}
		H = AA^T*BB
		U*S*Vt = singular_value_decomposition(H)
		R = U*Vt
		t = centroid_B-R*centroid_A
*/
Matrix4d ICP::best_fit_transform_SVD(const MatrixXd &A, const MatrixXd &B)
{
	size_t row = std::min(A.rows(), B.rows());

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

	Matrix4d T = MatrixXd::Identity(4,4);

	for(int i=0; i<row; i++)
	{
		centroid_A += A.block<1,3>(i,0).transpose();
		centroid_B += B.block<1,3>(i,0).transpose();
	}

	centroid_A /= row;
	centroid_B /= row;

	for(int i=0; i<row; i++)
	{
		AA.block<1,3>(i,0) = A.block<1,3>(i,0)-centroid_A.transpose();
		BB.block<1,3>(i,0) = B.block<1,3>(i,0)-centroid_B.transpose();
	}

	MatrixXd H = AA.transpose()*BB;
	MatrixXd U;
	VectorXd S;
	MatrixXd V;
	MatrixXd Vt;
	Matrix3d R;
	Vector3d t;
	
	JacobiSVD<MatrixXd> svd(H, ComputeFullU | ComputeFullV);
	// JacobiSVD decomposition computes only the singular values by default. 
	// ComputeFullU or ComputeFullV : ask for U or V explicitly.
	U = svd.matrixU();
	S = svd.singularValues();
	V = svd.matrixV();
	Vt = V.transpose();

    R = Vt.transpose()*U.transpose();

	if(R.determinant()<0)
	{
		Vt.block<1,3>(2,0) *= -1;
        R = Vt.transpose()*U.transpose();
	}

	t = centroid_B - R*centroid_A;

	T.block<3,3>(0,0) = R;
	T.block<3,1>(0,3) = t;

	std::cout << T << std::endl;

	return T;
}

Matrix4d ICP::best_fit_transform_quat(const MatrixXd &A, const MatrixXd &B)
{
	size_t row = std::min(A.rows(), B.rows());

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

	for(int i=0; i<row; i++)
	{
		centroid_A += A.block<1,3>(i,0).transpose();
		centroid_B += B.block<1,3>(i,0).transpose();
	}

	centroid_A /= row;
	centroid_B /= row;

	for(int i=0; i<row; i++)
	{
		AA.block<1,3>(i,0) = A.block<1,3>(i,0)-centroid_A.transpose();
		BB.block<1,3>(i,0) = B.block<1,3>(i,0)-centroid_B.transpose();
	}

	MatrixXd H = AA.transpose()*BB; // 3x3 covariance matrix
	MatrixXd As = H - H.transpose(); // anti-symmetric matrix

	Vector3d delta(As(1,2), As(2,0), As(0,1));

	MatrixXd Q = MatrixXd::Zero(4, 4);
	MatrixXd temp = MatrixXd::Zero(3, 3);
	MatrixXd traceEye = temp;
	double cov_trace = H.trace();

	for (int r = 0; r < traceEye.rows(); r++) 
	{
		for (int c = 0; c < traceEye.cols(); c++) 
		{
			if (r == c) 
			{
				traceEye(r, c) = cov_trace;
			}
		}
	}

	temp = H + H.transpose() - traceEye;

	Q(0, 0) = cov_trace;
	Q.block<1,3>(0,1) = delta.transpose();
	Q.block<3,1>(1,0) = delta;	

	Q.block<1,3>(1,1) = temp.block<1,3>(0,0);
	Q.block<1,3>(2,1) = temp.block<1,3>(1,0);
	Q.block<1,3>(3,1) = temp.block<1,3>(2,0);

	EigenSolver<MatrixXd> es(Q);
	VectorXd eVals = es.eigenvalues().real();
	MatrixXd eVecs = es.eigenvectors().real();

	// std::cout << "The eigenvalues of Q are:" << std::endl << eVals << std::endl;
	// std::cout << "The matrix of eigenvectors, V, is:" << std::endl << eVecs << std::endl << std::endl;

	// get location of maximum
  	Eigen::Index maxRow, maxCol;
  	// std::complex<double> max = eVals.maxCoeff(&maxRow, &maxCol);
	double max = eVals.maxCoeff(&maxRow, &maxCol);

	// std::cout << "The max eigenvalue: " << max << ", with position: " << maxRow << "," << maxCol << std::endl;

	VectorXd maxVecs = eVecs.col(maxRow);

	MatrixXd R = MatrixXd::Zero(3, 3);

	double q0 = maxVecs(0, 0);
	double q1 = maxVecs(1, 0);
	double q2 = maxVecs(2, 0);
	double q3 = maxVecs(3, 0);

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

	T.block<3,3>(0,0) = R;
	T.block<3,1>(0,3) = t;

	std::cout << T << std::endl;

	return T;
}

/*
	iterative closest point algorithm
	Input: 
		source      A = {a1,...,an}, ai = [x,y,z]
		destination B = {b1,...,bn}, bi = [x,y,z]
		max_iteration
		tolenrance
		outlierThreshold
	Output: 
		ICP_OUT->
			trans : transformation for best align
			dictances[i] : the distance between node i in src and its nearst node in dst
			inter : number of iterations
	Matrix:
		A = [x1,y1,z1]			B = [x1,y1,z1]
			|x2,y2,z2|				|x2,y2,z2|
			|........|				|........|
			[xn,yn,zn]				[xn,yn,zn]
		src = [x1,x2,x3, ...]		dst = [x1,x2,x3, ...]
			  |y1,y2,y3, ...|			  |y1,y2,y3, ...|
			  |z1,z2,z3, ...|			  |z1,z2,z3, ...|
			  [ 1, 1, 1, ...]			  [ 1, 1, 1, ...]
		* The last line is set for translation t, so that the T*src => M(3x4)*M(4*n)
		*  Notice that when src = T*src, the last line's maintain 1 and didn't be covered
		src3d = [x1,y1,z1]		
				|x2,y2,z2|		
				|........|		
				[xn,yn,zn]		
		* src3d : save the temp matrix transformed in this iteration
*/
ICP_OUT ICP::icp_alg(const MatrixXd &A, const MatrixXd &B, int max_iteration, float tolerance, int leaf_size, int Ksearch)
{
	size_t row = std::min(A.rows(),B.rows());
	MatrixXd src = MatrixXd::Ones(3+1,row);
	MatrixXd src3d = MatrixXd::Ones(3,row);
	MatrixXd dst = MatrixXd::Ones(3+1,row);
    NEIGHBOR neighbor;
  	Matrix4d T;
    Eigen::MatrixXd dst_chorder = Eigen::MatrixXd::Ones(3,row);
	ICP_OUT result;
	int iter;

	for(int i=0; i<row; i++)
	{
		src.block<3,1>(0,i) = A.block<1,3>(i,0).transpose(); // line 4 for t:translate
		src3d.block<3,1>(0,i) = A.block<1,3>(i,0).transpose(); // save the temp data
		dst.block<3,1>(0,i) = B.block<1,3>(i,0).transpose();
	}

	double prev_error = 0;
	double mean_error = 0;

	// When the number of iterations is less than the maximum
	for(iter=0; iter<max_iteration; iter++)
	{
        // neighbor = nearest_neighbor(src3d.transpose(), B);
		neighbor = nearest_neighbor_kdtree(src3d.transpose(), B);

        for(int j=0; j<row; j++)
		{
            dst_chorder.block<3,1>(0,j) = dst.block<3,1>(0, neighbor.indices[j]);
        }

		// save the transformed matrix in this iteration
		T = best_fit_transform_quat(src3d.transpose(), dst_chorder.transpose());
        // T = best_fit_transform_SVD(src3d.transpose(), dst_chorder.transpose());
		src = T*src;

		// copy the transformed matrix
		for(int j=0; j<row; j++)
        {
            src3d.block<3,1>(0,j) = src.block<3,1>(0,j);
        }

        mean_error = std::accumulate(neighbor.distances.begin(), neighbor.distances.end(),0.0)/neighbor.distances.size();
        std::cout << "error: " << prev_error - mean_error <<std::endl;
        if (abs(prev_error - mean_error) < tolerance)
        {
            break;
        }
        prev_error = mean_error;
	}

    T = best_fit_transform_quat(A, src3d.transpose());

	result.trans = T;
	result.distances = neighbor.distances;
	result.iter = iter+1;

	return result;
}

void ICP::align(pcl::PointCloud<pcl::PointXYZ>& cloud_icp_)
{
    MatrixXf source_matrix = cloud_icp->getMatrixXfMap(3,4,0).transpose();
    MatrixXf target_matrix = cloud_in->getMatrixXfMap(3,4,0).transpose();

    float tolerance = 0.000001;

    // call icp
    ICP_OUT icp_result = icp_alg(source_matrix.cast<double>(), target_matrix.cast<double>(), max_iter, tolerance);

    int iter = icp_result.iter;
    Matrix4f T = icp_result.trans.cast<float>();
    std::vector<float> distances = icp_result.distances;

    MatrixXf source_trans_matrix = source_matrix;
    
    int row = source_matrix.rows();
    MatrixXf source_trans4d = MatrixXf::Ones(3+1,row);
    for(int i=0; i<row; i++)
    {
        source_trans4d.block<3,1>(0,i) = source_matrix.block<1,3>(i,0).transpose();
    }
    source_trans4d = T*source_trans4d;
    for(int i=0; i<row; i++)
    {
        source_trans_matrix.block<1,3>(i,0)=source_trans4d.block<3,1>(0,i).transpose();
    }

    pcl::PointCloud<pcl::PointXYZ> temp_cloud;
    temp_cloud.width = row;
    temp_cloud.height = 1;
    temp_cloud.points.resize(row);
    for(size_t n=0; n<row; n++)
    {
        temp_cloud[n].x = source_trans_matrix(n, 0);
        temp_cloud[n].y = source_trans_matrix(n, 1);
        temp_cloud[n].z = source_trans_matrix(n, 2);	
    }
    cloud_icp_ = temp_cloud;
}

NEIGHBOR ICP::nearest_neighbor_naive(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst)
{
    int row_src = src.rows();
    int row_dst = dst.rows();
    Eigen::Vector3d vec_src;
    Eigen::Vector3d vec_dst;
    NEIGHBOR neigh;
    float min = 100;
    int index = 0;
    float dist_temp = 0;

    for(int ii=0; ii < row_src; ii++)
    {
        vec_src = src.block<1,3>(ii, 0).transpose();
        min = 100;
        index = 0;
        dist_temp = 0;
        for(int jj=0; jj < row_dst; jj++)
        {
            vec_dst = dst.block<1,3>(jj, 0).transpose();
            dist_temp = dist(vec_src, vec_dst);
            if(dist_temp < min)
            {
                min = dist_temp;
                index = jj;
            }
        }
        // cout << min << " " << index << endl;
        // neigh.distances[ii] = min;
        // neigh.indices[ii] = index;
        neigh.distances.push_back(min);
        neigh.indices.push_back(index);
    }

    return neigh;
}

NEIGHBOR ICP::nearest_neighbor_kdtree(const Eigen::MatrixXd &src, const Eigen::MatrixXd &dst)
{
    int row_dst = dst.rows();
    NEIGHBOR neigh;

	using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    matrix_t mat = src;

	using my_kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<matrix_t, SAMPLES_DIM /*fixed size*/, nanoflann::metric_L2>;
	my_kd_tree_t mat_index(SAMPLES_DIM, std::cref(mat), 10 /* max leaf */);

	for(int i = 0; i < row_dst; ++i)
	{
		std::vector<double> query_pt(SAMPLES_DIM);
		for(size_t d = 0; d < SAMPLES_DIM; d++)
		{
			query_pt[d] = dst(i,d);
		}

		// do a knn search
    	const size_t num_results = 1;
    	std::vector<size_t> ret_indexes(num_results);
    	std::vector<double> out_dists_sqr(num_results);

		nanoflann::KNNResultSet<double> resultSet(num_results);
		resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

		mat_index.index_->findNeighbors(resultSet, &query_pt[0]);

		// std::cout << "knnSearch(nn=" << num_results << "): \n";

		// for(size_t i = 0; i < resultSet.size(); i++)
		// {
		// 	std::cout << "ret_index[" << i << "]=" << ret_indexes[i]
		// 			<< " out_dist_sqr=" << out_dists_sqr[i] << std::endl;
		// }		

		neigh.distances.push_back(out_dists_sqr[0]);
        neigh.indices.push_back(ret_indexes[0]);
	}

	return neigh;
}