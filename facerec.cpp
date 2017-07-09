#include "facerec.h"
#include <iomanip>
#include <opencv2/highgui/highgui.hpp>
#include <unordered_map>
#include <iostream>
#include <fstream>


void FaceRec::normalize_vector(FV& fv) const {
	int n = fv.size();
	// get avg
	double sum = 0.0;
	for (int k = 0; k < n; ++k) {
		sum += std::pow(fv[k], 2);
	}
	sum = std::sqrt(sum);

	if (sum > 0.0) {
		for (int k = 0; k < n; ++k) {
			fv[k] /= sum;
		}
	}

}

std::vector<double> FaceRec::create_feature_vector(const cv::Mat& img) const {
	std::vector<double> feature_vector(img.rows * img.cols);
	for (int row = 0; row < img.rows; ++row) {
		for (int col = 0; col < img.cols; ++col) {
			feature_vector[row * img.cols + col] = img.at<unsigned char>(row, col);
		}
	}
	return feature_vector;
}

std::vector<FV> FaceRec::create_feature_vectors(const std::vector<cv::Mat>& imgs) const {
	std::vector<FV> fvs(imgs.size());
	for (int i = 0; i < imgs.size(); ++i) {
		fvs[i] = create_feature_vector(imgs[i]);
	}
	return fvs;
}

void FaceRec::create_feature_vectors(const std::unordered_map<int, Imgs>& imgs_by_class,
	std::unordered_map<int, std::vector<FV>>& fvs_by_class) const {
	
	for (auto& classes : imgs_by_class) {
		std::vector<FV> fvs;
		for (auto& img : classes.second) {
			auto fv = create_feature_vector(img);
			fvs.push_back(fv);
		}
		fvs_by_class[classes.first] = fvs;
	}
}

FV FaceRec::calculate_mean(std::vector<std::vector<double>>& feature_vectors) const {
	int N = feature_vectors.size();

	std::vector<double> mean_vector;
	if (feature_vectors.size() < 1) {
		return  mean_vector;
	}

	int dim = feature_vectors[0].size();
	mean_vector.resize(dim);
	std::fill(mean_vector.begin(), mean_vector.end(), 0.0);

	for (int i = 0; i < feature_vectors.size(); ++i) {
		for (int k = 0; k < dim; ++k) {
			mean_vector[k] += feature_vectors[i][k];
		}
	}

	// get avg
	for (int k = 0; k < dim; ++k) {
		mean_vector[k] /= N;
	}

	return mean_vector;
}

FV FaceRec::calculate_mean(const std::unordered_map<int, std::vector<FV>>& fvs_by_class) const {

	std::vector<double> mean_vector;

	int dim = 0;
	for (auto& classes : fvs_by_class) {
		for (auto& fv : classes.second) {
			dim = fv.size();
			break;
		}
	}

	//if (feature_vectors.size() < 1) {
	//	return  mean_vector;
	//}

	mean_vector.resize(dim);
	std::fill(mean_vector.begin(), mean_vector.end(), 0.0);

	int N = 0;
	for (auto& classes : fvs_by_class) {
		for (auto& fv : classes.second) {
			for (int k = 0; k < dim; ++k) {
				mean_vector[k] += fv[k];
			}
			N++;
		}
	}

	// get avg
	for (int k = 0; k < dim; ++k) {
		mean_vector[k] /= N;
	}

	return mean_vector;
}

FV FaceRec::calculate_mean(std::vector<cv::Mat>& imgs) {

	std::vector<FV> fvs(imgs.size());
	for (int i = 0; i < imgs.size(); ++i) {
		fvs[i] = create_feature_vector(imgs[i]);
	}

	FV mean_vec = calculate_mean(fvs);
	return mean_vec;

}

double FaceRec::norm(const FV& fv) const {
	double sum = 0.0;
	for (auto& f : fv) {
		sum += std::pow(f, 2);
	}
	return std::sqrt(sum);
}

FV operator+(const FV& left, const FV& right) {
	assert(left.size() == right.size());
	FV added(left.size());
	for (size_t i = 0; i < left.size(); ++i) {
		added[i] = left[i] + right[i];
	}
	return added;
}

FV operator*(double scalar, const FV& fv) {
	FV mult(fv.size());
	for (size_t i = 0; i < fv.size(); ++i) {
		mult[i] = scalar * fv[i];
	}
	return mult;
}

FV operator-(const FV& left, const FV& right) {
	assert(left.size() == right.size());
	FV subt(left.size());
	return left + -1.0 * right;
}

cv::Mat FaceRec::calc_X(const std::unordered_map<int, Imgs>& imgs_by_class, 
	std::unordered_map<int, std::vector<FV>>& fvs_by_class) const {

	create_feature_vectors(imgs_by_class, fvs_by_class);
	auto mean_vec = calculate_mean(fvs_by_class);

	int N = 0;
	for (auto& classes : fvs_by_class) {
		for (auto& fv : classes.second) {
			N++;
		}
	}

	cv::Mat X(mean_vec.size(), N, CV_64F);

	int i = 0;
	for (auto& classes : fvs_by_class) {
		for (auto& fv : classes.second) {
			auto mean_subtracted_fv = fv - mean_vec;
			normalize_vector(mean_subtracted_fv);
			fv = mean_subtracted_fv;
			cv::Mat col(fv.size(), 1, CV_64F, &mean_subtracted_fv[0]);
			col.copyTo(X.col(i));
			i++;
		}
	}

	//C /= fvs.size();
	return X;
}

void FaceRec::subtract_mean_and_normalize(const FV& mean, std::unordered_map<int, std::vector<FV>>& fvs_by_class) const {
	int i = 0;
	for (auto& classes : fvs_by_class) {
		for (auto& fv : classes.second) {
			auto mean_subtracted_fv = fv - mean;
			normalize_vector(mean_subtracted_fv);
			fv = mean_subtracted_fv;
			i++;
		}
	}
	
}

void FaceRec::calc_efficient_eigen_values(const cv::Mat& X, cv::Mat& w, int& eigen_vals) const {

	cv::Mat E, V;
	cv::Mat XtX = X.t() * X;
	cv::eigen(XtX, E, V);

	for (int i = 0; i < E.rows; ++i) {
		if (E.at<double>(i, 0) < 1.0) {
			eigen_vals = i;
			break;
		}
	}

	// already sorted, no need to sort
	cv::Mat w_unnormalized = X * V.t();

	w = cv::Mat::zeros(w_unnormalized.size(), w_unnormalized.type());

	cv::Range rr(0, w.rows - 1);
	// normalize
	for (int i = 0; i < w_unnormalized.cols; ++i) {
		cv::Mat normalized_col = w_unnormalized.col(i) / cv::norm(w_unnormalized.col(i));
		normalized_col.copyTo(w.col(i));
	}
}

void FaceRec::classify_imgs(const Imgs& training, const Imgs& testing) const {

}

void FaceRec::calculate_class_means(const std::unordered_map<int, std::vector<FV>>& fvs_by_class, std::unordered_map<int, FV>& mean_by_class) const {

	int dim = 0;
	for (auto& classes : fvs_by_class) {
		for (auto& fv : classes.second) {
			dim = fv.size();
			break;
		}
	}


	for (auto& classes : fvs_by_class) {
		int N = 0;
		std::vector<double> mean_vector(dim);
		std::fill(mean_vector.begin(), mean_vector.end(), 0.0);
		for (auto& fv : classes.second) {
			for (int k = 0; k < dim; ++k) {
				mean_vector[k] += fv[k];
			}
			N++;
		}
		// get avg
		for (int k = 0; k < dim; ++k) {
			mean_vector[k] /= N;
		}
		mean_by_class[classes.first] = mean_vector;
	}

}

cv::Mat FaceRec::calc_Z(const FV& mean, std::unordered_map<int, FV>& mean_by_class, const std::unordered_map<int, std::vector<FV>>& fvs_by_class) const {
	// build mean_i - mean

	cv::Mat mi_m(mean.size(), mean_by_class.size(), CV_64F);

	int i = 0;
	for (auto& mean_fv_struct : mean_by_class) {
		auto& mean_fv = mean_fv_struct.second;
		auto mean_i_minus_mean = mean_fv - mean;
		//normalize_vector(mean_i_minus_mean);
		cv::Mat col(mean.size(), 1, CV_64F, &mean_i_minus_mean[0]);
		col.copyTo(mi_m.col(i));
		i++;
	}


	cv::Mat E, transpose_V;
	cv::Mat mimtmim = mi_m.t() * mi_m;
	cv::eigen(mimtmim, true, E, transpose_V);

	int no_of_features = 0;

	for (int k = 0; k < E.rows; ++k) {
		if (E.at<double>(k, 0) < 1) {
			no_of_features = k;
			break;
		}
	}

	//cv::Mat D = E(cv::Rect(0, 0, 1, no_of_features));

	cv::Mat V = mi_m * transpose_V.t();


	cv::Mat Y = V(cv::Rect(0, 0, no_of_features ,V.rows));

	cv::Mat test = Y.t() * Y;
	// E holds the eigen values for mi_m * mi_m' as well

	//cv::Mat Db = cv::Mat::diag(D / mean_by_class.size());

	//cv::Mat Db_sqrt;
	//cv::pow(D, -0.5, Db_sqrt);




	////cv::Mat t = V.t() * V;
	//cv::Mat SB_V_unnormalized = mi_m * V;

	//cv::Mat SB_V(SB_V_unnormalized.size(), SB_V_unnormalized.type());
	//for (int k = 0; k < SB_V_unnormalized.cols; ++k) {
	//	cv::Mat normalkzed_col = SB_V_unnormalized.col(k) / cv::norm(SB_V_unnormalized.col(k));
	//	normalkzed_col.copyTo(SB_V.col(k));
	//}
	////for (int k = 0; k < SB_V_unnormalized.rows; ++k) {
	////	cv::Mat normalkzed_col = SB_V_unnormalized.row(k) / cv::norm(SB_V_unnormalized.row(k));
	////	normalkzed_col.copyTo(SB_V.row(k));
	////}

	//cv::Mat t2 = SB_V.t() * SB_V;


	//cv::Mat Y = SB_V(cv::Rect(0, 0, SB_V.cols, SB_V.rows));


	cv::Mat DB_unclean = Y.t() * mi_m  * mi_m.t() * Y;
	cv::Mat DB_diag = DB_unclean.diag();
	cv::Mat DB_minus_0_5;
	cv::pow(DB_diag, -0.5, DB_minus_0_5);

	cv::Mat DB_minus_0_5_full = cv::Mat::diag(DB_minus_0_5);

	//cv::Mat DB_inv = DB.inv();
	cv::Mat Z = Y * DB_minus_0_5_full;

	return Z;
	//cv::Mat Z = Y * Db_sqrt;

	//int dim = 0;
	//int no_of_imgs = 0;
	//for (auto& classes : fvs_by_class) {
	//	no_of_imgs = classes.second.size();
	//	for (auto& fv : classes.second) {
	//		dim = fv.size();
	//		break;
	//	}
	//}

	//cv::Mat xk_mi(dim, mean_by_class.size() * no_of_imgs, CV_64F);

	//cv::Mat Zt_SW_Z = cv::Mat::zeros(no_of_features, no_of_features, CV_64F);

	//i = 0;
	//for (auto& classes : fvs_by_class) {
	//	cv::Mat tmp = cv::Mat::zeros(no_of_features, no_of_features, CV_64F);
	//	for (auto& fv : classes.second) {
	//		auto result = mean_by_class.find(classes.first);
	//		auto mean_class_subtracted_fv = fv - result->second;
	//		cv::Mat col(fv.size(), 1, CV_64F, &mean_class_subtracted_fv[0]);
	//		cv::Mat X = Z.t() * col;
	//		tmp += X * X.t();
	//		i++;
	//	}
	//	Zt_SW_Z += tmp / classes.second.size();
	//}
	//Zt_SW_Z /= fvs_by_class.size();

	//cv::Mat E2, U;
	//cv::eigen(Zt_SW_Z, E2, U);

	//// we need to keep the smallest eigen vectors

	//return U;
}

cv::Mat FaceRec::calc_eigen_vector_Z_Sw(const cv::Mat& Z, const std::unordered_map<int, std::vector<FV>>& fvs_by_class, 
	const std::unordered_map<int, FV>& mean_by_class) const {

	int dim = 0;
	int no_of_imgs = 0;
	for (auto& classes : fvs_by_class) {
		no_of_imgs = classes.second.size();
		for (auto& fv : classes.second) {
			dim = fv.size();
			break;
		}
	}

	cv::Mat xk_mi(dim, mean_by_class.size() * no_of_imgs, CV_64F);

	//cv::Mat Zt_SW_Z = cv::Mat::zeros(no_of_features, no_of_features, CV_64F);

	int i = 0;
	for (auto& classes : fvs_by_class) {
		//cv::Mat tmp = cv::Mat::zeros(no_of_features, no_of_features, CV_64F);
		for (auto& fv : classes.second) {
			auto result = mean_by_class.find(classes.first);
			auto mean_class_subtracted_fv = fv - result->second;
			cv::Mat col(fv.size(), 1, CV_64F, &mean_class_subtracted_fv[0]);
			col.copyTo(xk_mi.col(i));
			//cv::Mat X = Z.t() * col;
			//tmp += X * X.t();
			i++;
		}
		//Zt_SW_Z += tmp / classes.second.size();
	}
	//Zt_SW_Z /= fvs_by_class.size();

	cv::Mat Zt_xk_mi = Z.t() * xk_mi;

	cv::Mat Zt_Sw_Z = Zt_xk_mi * Zt_xk_mi.t();

	cv::Mat E2, V_transpose;
	cv::eigen(Zt_Sw_Z, E2, V_transpose);

	// we need to keep the smallest eigen vectors
	// so, reverse cols, opencv has it in descending order
	cv::Mat V_transpose_reversed = V_transpose.clone();
	for (int k = 0; k < V_transpose.rows; ++k) {
		V_transpose.row(k).copyTo(V_transpose_reversed.row(V_transpose_reversed.rows - 1 - k));
	}

	cv::Mat U = V_transpose_reversed.t();


	return U;
}

FaceRec::FaceRec()
{
}


FaceRec::~FaceRec()
{
}

int main(int argc, char** argv) {

	int no_of_people = 30;
	int no_of_imgs_per_person = 21;

	//int no_of_people = 5;
	//int no_of_imgs_per_person = 5;

	std::unordered_map<int, Imgs> training_imgs_by_class;
	std::unordered_map<int, Imgs> testing_imgs_by_class;
	//Imgs training;
	//Imgs testing;

	std::string train_prefix("train\\");
	std::string test_prefix("test\\");

	FaceRec face_rec;
	for (int p = 0; p < no_of_people; ++p) {
		for (int i = 0; i < no_of_imgs_per_person; ++i) {
			std::stringstream train_ss;
			train_ss << std::setfill('0') << std::setw(2);
			train_ss <<  train_prefix << std::setw(2) << p + 1 << "_" << std::setw(2) << i + 1 << ".png";
			std::stringstream test_ss;
			test_ss << std::setfill('0') << std::setw(2);
			test_ss <<  test_prefix << std::setw(2) << p + 1 << "_" << std::setw(2) << i + 1 << ".png";

			std::string filename = train_ss.str();
			cv::Mat traing_img = cv::imread(filename, 0);
			FV fv = face_rec.create_feature_vector(traing_img);

			training_imgs_by_class[p+1].push_back(traing_img);
			//training.push_back(traing_img);

			cv::Mat test_img = cv::imread(test_ss.str(), 0);
			testing_imgs_by_class[p + 1].push_back(test_img);
			//testing.push_back(test_img);
		}
	}

	///*
	// LDA Output
	std::unordered_map<int, std::vector<FV>> training_normalized_fv_by_class;
	std::unordered_map<int, std::vector<FV>> testing_normalized_fv_by_class;


	face_rec.create_feature_vectors(training_imgs_by_class, training_normalized_fv_by_class);
	FV training_mean_fv = face_rec.calculate_mean(training_normalized_fv_by_class);
	//face_rec.subtract_mean_and_normalize(training_mean_fv, training_normalized_fv_by_class);

	face_rec.create_feature_vectors(testing_imgs_by_class, testing_normalized_fv_by_class);
	FV testing_mean_fv = face_rec.calculate_mean(testing_normalized_fv_by_class);
	//face_rec.subtract_mean_and_normalize(testing_mean_fv, testing_normalized_fv_by_class);

	std::unordered_map<int, FV> class_means;
	face_rec.calculate_class_means(training_normalized_fv_by_class, class_means);
	cv::Mat Z = face_rec.calc_Z(training_mean_fv, class_means, training_normalized_fv_by_class);

	cv::Mat U = face_rec.calc_eigen_vector_Z_Sw(Z, training_normalized_fv_by_class, class_means);

	//int no_of_features = 3;

	std::ofstream file("LDA.csv");
	std::unordered_map<int, Imgs> projected_training_imgs_by_class;
	std::unordered_map<int, Imgs> projected_testing_imgs_by_class;

	int total = 0;
	int positive = 0;
	// project to eigen space, and test

	for (int i = 0; i < U.cols; ++i) {
		cv::Mat trunc_U = U(cv::Rect(0, 0, i+1, U.rows));
		cv::Mat trunc_w_unnormalized = Z * trunc_U;

		cv::Mat trunc_w(trunc_w_unnormalized.size(), trunc_w_unnormalized.type());
		// normalize
		for (int k = 0; k < trunc_w_unnormalized.cols; ++k) {
			cv::Mat normalkzed_col = trunc_w_unnormalized.col(k) / cv::norm(trunc_w_unnormalized.col(k));
			normalkzed_col.copyTo(trunc_w.col(k));
		}

		// project all training imgs
		for (auto& classes : training_normalized_fv_by_class) {
			Imgs projected_imgs;
			for (auto& fv : classes.second) {
				auto fv_mean = fv - training_mean_fv;
				cv::Mat fv_mat(fv.size(), 1, CV_64F, &fv_mean[0]);
				cv::Mat proj_img = trunc_w.t() * fv_mat;
				projected_imgs.push_back(proj_img);
			}
			projected_training_imgs_by_class[classes.first] = projected_imgs;
		}

		// project all testing imgs
		for (auto& classes : testing_normalized_fv_by_class) {
			Imgs projected_imgs;
			for (auto& fv : classes.second) {
				auto fv_mean = fv - training_mean_fv;
				cv::Mat fv_mat(fv.size(), 1, CV_64F, &fv_mean[0]);
				cv::Mat proj_img = trunc_w.t() * fv_mat;
				projected_imgs.push_back(proj_img);
			}
			projected_testing_imgs_by_class[classes.first] = projected_imgs;
		}


		// classify with NN
		struct NNData {
			int i;
			double dist;
		};


		for (auto& testing_classes : projected_testing_imgs_by_class) {
			int class_total = 0;
			int class_positive = 0;
			for (auto& testing_projection : testing_classes.second) {
				
				std::vector<NNData> nn_vector;
				// check with all training
				for (auto& training_classes : projected_training_imgs_by_class) {
					for (auto& training_projection : training_classes.second) {
						cv::Mat diff = testing_projection - training_projection;
						double l2norm = cv::norm(diff);
						NNData data;
						data.i = training_classes.first;
						data.dist = l2norm;
						nn_vector.push_back(data);
					}
				}

				std::sort(nn_vector.begin(), nn_vector.end(), [&](const NNData& lhs, const NNData& rhs)
				{
					return lhs.dist < rhs.dist;
				});

				// get nearest as result
				total++;
				class_total++;
				if (nn_vector[0].i == testing_classes.first) {
					positive++;
					class_positive++;
				}

			}
			//std::cout << "Class Accuracy : " << class_positive / (double)(class_total) << "\n";
		}

		double avg = positive / (double)(total);
		std::cout << " Eigen vals : " << i + 1 << " Accuracy : " << avg << "\n";
		file << (i + 1) << "," << avg << "\n";


	}
//*/

	 //PCA Output
	std::unordered_map<int, std::vector<FV>> training_normalized_fv_by_class;
	std::unordered_map<int, std::vector<FV>> testing_normalized_fv_by_class;

	cv::Mat X = face_rec.calc_X(training_imgs_by_class, training_normalized_fv_by_class);

	// repeat same to normalize test
	face_rec.calc_X(testing_imgs_by_class, testing_normalized_fv_by_class);
	cv::Mat w;
	int significant_eigen_vals;
	face_rec.calc_efficient_eigen_values(X, w, significant_eigen_vals);

	std::unordered_map<int, Imgs> projected_training_imgs_by_class;
	std::unordered_map<int, Imgs> projected_testing_imgs_by_class;

	int total = 0;
	int positive = 0;
	// project to eigen space, and test

	std::vector<double> accuracies;
	std::ofstream file("PCA.csv");

	for (int i = 0; i < significant_eigen_vals; ++i) {
		cv::Mat trunc_w = w(cv::Rect(0, 0, i+1, w.rows));

		// project all training imgs
		for (auto& classes : training_normalized_fv_by_class) {
			Imgs projected_imgs;
			for (auto& fv : classes.second) {
				cv::Mat fv_mat(fv.size(), 1, CV_64F, &fv[0]);
				cv::Mat proj_img = trunc_w.t() * fv_mat;
				projected_imgs.push_back(proj_img);
			}
			projected_training_imgs_by_class[classes.first] = projected_imgs;
		}

		// project all testing imgs
		for (auto& classes : testing_normalized_fv_by_class) {
			Imgs projected_imgs;
			for (auto& fv : classes.second) {
				cv::Mat fv_mat(fv.size(), 1, CV_64F, &fv[0]);
				cv::Mat proj_img = trunc_w.t() * fv_mat;
				projected_imgs.push_back(proj_img);
			}
			projected_testing_imgs_by_class[classes.first] = projected_imgs;
		}


		// classify with NN
		struct NNData {
			int i;
			double dist;
		};


		for (auto& testing_classes : projected_testing_imgs_by_class) {
			int class_total = 0;
			int class_positive = 0;
			for (auto& testing_projection : testing_classes.second) {
				
				std::vector<NNData> nn_vector;
				// check with all training
				for (auto& training_classes : projected_training_imgs_by_class) {
					for (auto& training_projection : training_classes.second) {
						cv::Mat diff = testing_projection - training_projection;
						double l2norm = cv::norm(diff);
						NNData data;
						data.i = training_classes.first;
						data.dist = l2norm;
						nn_vector.push_back(data);
					}
				}

				std::sort(nn_vector.begin(), nn_vector.end(), [&](const NNData& lhs, const NNData& rhs)
				{
					return lhs.dist < rhs.dist;
				});

				// get nearest as result
				total++;
				class_total++;
				if (nn_vector[0].i == testing_classes.first) {
					positive++;
					class_positive++;
				}

			}
			//std::cout << "Class Accuracy : " << class_positive / (double)(class_total) << "\n";
		}

		double avg = positive / (double)(total);
		std::cout << " Eigen vals : " << i + 1 << " Accuracy : " << avg << "\n";
		file << (i + 1) << "," << avg << "\n";
		accuracies.push_back(avg);
	}
	 //*/


}