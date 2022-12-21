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

int cmpKNeighbor(const void *a, const void *b)
{
	KNeighbor *a1 = (KNeighbor*)a;
	KNeighbor *a2 = (KNeighbor*)b;
	return (*a1).distanceMean - (*a2).distanceMean;
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
Matrix4d ICP::best_fit_transform(const MatrixXd &A, const MatrixXd &B)
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

	// R = U*Vt;
    R = Vt.transpose()*U.transpose();

	if(R.determinant()<0)
	{
		Vt.block<1,3>(2,0) *= -1;
		// R = U*Vt;
        R = Vt.transpose()*U.transpose();
	}

	t = centroid_B - R*centroid_A;

	T.block<3,3>(0,0) = R;
	T.block<3,1>(0,3) = t;
	return T;
}

/*
	Input : A : n*3 matrix
			B : n*3 matrix
    	    neighbors : Indexes and distances of k closest points match.
		    remainPercentage x = [0 ~ 100] : Remove worst (100-x)% of 
		    					correspondence for outlier rejection. 
*/
Matrix4d ICP::best_fit_transform(const MatrixXd &A, const MatrixXd &B, std::vector<KNeighbor> neighbors, int remainPercentage, int K)
{
	int num = (int) neighbors.size()*remainPercentage/100;
	Vector3d centroid_A(0,0,0);
	Vector3d centroid_B(0,0,0);
	Vector3d temp(0,0,0);
	MatrixXd AA = A;
	MatrixXd BB = A;
    // MatrixXd BB = B;
	Matrix4d T = MatrixXd::Identity(4,4);

	for(int i=0; i<num; i++)
	{
		int aIndex = neighbors[i].sourceIndex;
		centroid_A += A.block<1,3>(aIndex, 0).transpose();
		for(int k=0; k<K; k++)
		{
			int bIndex = neighbors[i].targetIndexes[k];
			centroid_B += B.block<1,3>(bIndex, 0).transpose();
		}
		centroid_B /= K;
	}

	centroid_A /= num;
	centroid_B /= num;

	for(int i=0; i<num; i++)
	{
		int aIndex = neighbors[i].sourceIndex;
		AA.block<1,3>(i,0) = A.block<1,3>(aIndex, 0)-centroid_A.transpose();

		for(int k=0; k<K; k++)
		{
			int bIndex = neighbors[i].targetIndexes[k];
			BB.block<1,3>(i,0) = B.block<1,3>(bIndex, 0)-centroid_B.transpose();
		}
		BB.block<1,3>(i,0) /= K;	
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

	R = U*Vt;

	if(R.determinant()<0)
	{
		Vt.block<1,3>(2,0) *= -1;
		R = U*Vt;
	}

	t = centroid_B-R*centroid_A;

	T.block<3,3>(0,0) = R;
	T.block<3,1>(0,3) = t;
	return T;
}

std::vector<KNeighbor> ICP::k_nearest_neighbors(const MatrixXd& source, const MatrixXd& target, float leaf_size, int K)
{
	int dimension = 3;
	int SourceRow = source.rows();
	int targetRow = target.rows();
	Vector3d sourceVector;
	Vector3d targetVector;
	std::vector<KNeighbor> neighbors;
	int tempIndex = 0;
	float tempDistance = 0;

	// build kd-tree
	Matrix<float, Dynamic, Dynamic> targetMatrix(targetRow, dimension);
	for(int i=0; i<targetRow; i++)
    {
        for(int d=0; d<dimension; d++)
        {
            targetMatrix(i,d) = target(i,d);
        }
    }

	typedef KDTreeEigenMatrixAdaptor<Matrix<float,Dynamic,Dynamic>> kdtree_t;
	kdtree_t targetKDtree(dimension, std::cref(targetMatrix), leaf_size);
	targetKDtree.index_->buildIndex();

	for(int i=0; i<SourceRow; i++)
	{
		std::vector<float> sourcePoint(dimension);
		float meanDis = 0.0f;

		for(size_t d=0; d<dimension; d++)
        {
            sourcePoint[d]=source(i, d);
        }			

		std::vector<size_t> result_indexes(K);
		std::vector<float> result_distances(K);
		nanoflann::KNNResultSet<float> resultSet(K);
		resultSet.init(&result_indexes[0], &result_distances[0]);
		nanoflann::SearchParameters params_igonored;
		targetKDtree.index_->findNeighbors(resultSet, &sourcePoint[0], params_igonored);

		KNeighbor neigh;
		neigh.sourceIndex = i;
		for(int j=0; j<K; j++)
		{
			neigh.targetIndexes.push_back(result_indexes[j]);
			neigh.distances.push_back(result_distances[j]);
			meanDis += result_distances[j];
		}
		neigh.distanceMean = meanDis/K;
		neighbors.push_back(neigh);
	}

	qsort(&neighbors[0], neighbors.size(), sizeof(KNeighbor), cmpKNeighbor);
	return neighbors;
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
    std::vector<KNeighbor> neighbors;
  	Matrix4d T;
  	// Matrix4d T_all = MatrixXd::Identity(4, 4);
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
		neighbors = k_nearest_neighbors(src3d.transpose(), B); // n*3,n*3

		// save the transformed matrix in this iteration
		T = best_fit_transform(src3d.transpose(), dst.transpose(), neighbors);
		// T_all = T*T_all;
		src = T*src; // notice the order of matrix product

		// copy the transformed matrix
		for(int j=0; j<row; j++)
        {
            src3d.block<3,1>(0,j) = src.block<3,1>(0,j);
        }

		// calculate the mean error
		mean_error = 0.0f;
		for(int i=0; i<neighbors.size(); i++)
        {
            mean_error += neighbors[i].distanceMean;
        }			
		mean_error /= neighbors.size();
		std::cout << "error: " << prev_error-mean_error <<std::endl;
		if(abs(prev_error - mean_error)<tolerance)
        {
            break;
        }
	
		prev_error = mean_error;
	}

	std::vector<float> distances;
	for(int i=0; i<neighbors.size(); i++)
    {
        distances.push_back(neighbors[i].distanceMean);
    }

    T = best_fit_transform(A, src3d.transpose());

	result.trans = T;
	result.distances = distances;
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
    // cloud_source_trans = temp_cloud.makeShared();
    // cloud_icp = temp_cloud.makeShared();
    cloud_icp_ = temp_cloud;
    // return temp_cloud;
}