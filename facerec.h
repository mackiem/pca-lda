#pragma once
#include <vector>
#include "opencv2/core/core.hpp"
#include <unordered_map>

using FV = std::vector < double > ;
using Imgs = std::vector < cv::Mat > ;

class FaceRec
{
public:
	// common
	void normalize_vector(FV& fv) const;
	FV create_feature_vector(const cv::Mat& img) const;
	std::vector<FV> create_feature_vectors(const std::vector<cv::Mat>& imgs) const;
	FV calculate_mean(std::vector<FV>& fvs) const;
	FV calculate_mean(const std::unordered_map<int, std::vector<FV>>& fvs_by_class) const;
	FV calculate_mean(std::vector<cv::Mat>& imgs);

	double norm(const FV& fv) const;

	// PCA
	cv::Mat FaceRec::calc_X(const std::unordered_map<int, Imgs>& imgs_by_class, std::unordered_map<int, std::vector<FV>>& fvs_by_class) const;
	void subtract_mean_and_normalize(const FV& mean, std::unordered_map<int, std::vector<FV>>& fvs_by_class) const;
	void FaceRec::create_feature_vectors(const std::unordered_map<int, Imgs>& imgs_by_class,
		std::unordered_map<int, std::vector<FV>>& fvs_by_class) const;

	//cv::Mat calc_X(const std::vector<cv::Mat>& imgs) const;
	void FaceRec::calc_efficient_eigen_values(const cv::Mat& X, cv::Mat& w, int& significant_eigen_vals) const;
	void classify_imgs(const Imgs& training, const Imgs& testing) const;

	//LDA
	void calculate_class_means(const std::unordered_map<int, std::vector<FV>>& fvs_by_class, std::unordered_map<int, FV>& mean_by_class) const;
	cv::Mat calc_Z(const FV& mean, std::unordered_map<int, FV>& mean_by_class, const std::unordered_map<int, std::vector<FV>>& fvs_by_class) const;
	//void calc_eigen_vector_Z_Sw(const cv::Mat& Z, const std::unordered_map<int, std::vector<FV>>& fvs_by_class, const std::unordered_map<int, FV>& mean_by_class,
	//	cv::Mat& U) const;
	cv::Mat FaceRec::calc_eigen_vector_Z_Sw(const cv::Mat& Z, const std::unordered_map<int, std::vector<FV>>& fvs_by_class,
		const std::unordered_map<int, FV>& mean_by_class) const;


	FaceRec();
	~FaceRec();
};

